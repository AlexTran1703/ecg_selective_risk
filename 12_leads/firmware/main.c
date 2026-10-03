/* On-target inference timing for the STM32F411.
 *
 * Minimal bare-metal application: bring the core to 100 MHz, run the
 * generated int8 network N times on a fixed input, and time each run with
 * the DWT cycle counter. Results are parked in a fixed SRAM block that
 * the host reads back over SWD, so no UART wiring is needed.
 *
 * Timing is in core cycles rather than milliseconds, because cycles are
 * what the part actually guarantees; the host converts using the clock
 * frequency this file sets, which is also written into the result block
 * so the two can never disagree.
 */

#include <stdint.h>
#include <string.h>

#include "network.h"
#include "network_data.h"

/* ---------------------------------------------------------- registers */
#define RCC_BASE      0x40023800UL
#define RCC_CR        (*(volatile uint32_t *)(RCC_BASE + 0x00))
#define RCC_PLLCFGR   (*(volatile uint32_t *)(RCC_BASE + 0x04))
#define RCC_CFGR      (*(volatile uint32_t *)(RCC_BASE + 0x08))
#define RCC_APB1ENR   (*(volatile uint32_t *)(RCC_BASE + 0x40))

#define FLASH_ACR     (*(volatile uint32_t *)0x40023C00UL)
#define PWR_CR        (*(volatile uint32_t *)0x40007000UL)

#define DEMCR         (*(volatile uint32_t *)0xE000EDFCUL)
#define DWT_CTRL      (*(volatile uint32_t *)0xE0001000UL)
#define DWT_CYCCNT    (*(volatile uint32_t *)0xE0001004UL)
#define DWT_LAR       (*(volatile uint32_t *)0xE0001FB0UL)

#define HSI_HZ        16000000UL
#define CORE_HZ       100000000UL
#define N_RUNS        32U
#define RESULT_MAGIC  0xEC61BEEFUL   /* must match s8_latency.py MAGIC */

/* Written by the firmware, read back by the host. Placed at a fixed
 * address by the linker script so the host does not need the ELF. */
typedef struct {
    uint32_t magic;
    uint32_t stage;
    uint32_t core_hz;
    uint32_t n_runs;
    int32_t  ai_error;        /* 0 = network created and ran cleanly */
    uint32_t macc_hint;
    uint32_t cycles[N_RUNS];
} results_t;

volatile results_t g_results __attribute__((section(".results"), used));

/* The arena the network allocates its activations in. */
static uint8_t g_activations[AI_NETWORK_DATA_ACTIVATIONS_SIZE]
    __attribute__((aligned(32)));
static int8_t g_input[AI_NETWORK_IN_1_SIZE_BYTES] __attribute__((aligned(8)));
static int8_t g_output[AI_NETWORK_OUT_1_SIZE_BYTES] __attribute__((aligned(8)));

/* ------------------------------------------------------------- clocks */
static void clock_100mhz(void)
{
    /* Flash must be slowed before the core speeds up, never after. */
    FLASH_ACR = (1U << 8) | (1U << 9) | (1U << 10) | 3U;   /* PRFT/IC/DC, 3WS */

    RCC_APB1ENR |= (1U << 28);                             /* PWREN */
    PWR_CR |= (3U << 14);                                  /* VOS scale 1 */

    RCC_CR |= (1U << 0);                                   /* HSION */
    while (!(RCC_CR & (1U << 1))) { }                      /* HSIRDY */

    RCC_CR &= ~(1U << 24);                                 /* PLLOFF */
    while (RCC_CR & (1U << 25)) { }

    /* 16 MHz / M16 = 1 MHz; x N400 = 400 MHz VCO; / P4 = 100 MHz. */
    RCC_PLLCFGR = 16U | (400U << 6) | (1U << 16) | (8U << 24);

    RCC_CR |= (1U << 24);                                  /* PLLON */
    while (!(RCC_CR & (1U << 25))) { }                     /* PLLRDY */

    RCC_CFGR = (4U << 10);                                 /* APB1 = /2 */
    RCC_CFGR |= 2U;                                        /* SW = PLL */
    while (((RCC_CFGR >> 2) & 3U) != 2U) { }               /* SWS = PLL */
}

static void dwt_init(void)
{
    DEMCR |= (1U << 24);            /* TRCENA */
    DWT_LAR = 0xC5ACCE55UL;         /* unlock, harmless where not needed */
    DWT_CYCCNT = 0U;
    DWT_CTRL |= 1U;                 /* CYCCNTENA */
}

/* --------------------------------------------------------------- main */
int main(void)
{
    ai_handle net = AI_HANDLE_NULL;
    const ai_handle acts[] = { (ai_handle)g_activations };
    ai_buffer *in, *out;
    ai_error err;
    uint32_t i, t0, t1;

    clock_100mhz();
    dwt_init();

    g_results.magic = RESULT_MAGIC;   /* written first: proves we ran */
    g_results.stage = 1U;
    g_results.core_hz = CORE_HZ;
    g_results.n_runs = N_RUNS;
    g_results.ai_error = -1;

    g_results.stage = 2U;
    err = ai_network_create_and_init(&net, acts, NULL);
    if (err.type != AI_ERROR_NONE) {
        g_results.ai_error = (int32_t)((err.type << 8) | err.code);
        g_results.stage = 99U;
        for (;;) { }
    }

    g_results.stage = 3U;
    in = ai_network_inputs_get(net, NULL);
    out = ai_network_outputs_get(net, NULL);

    /* A fixed, non-trivial input. The values do not change the work the
     * network does, only which branch of a max() wins, so a deterministic
     * ramp is enough and keeps the measurement reproducible. */
    for (i = 0; i < (uint32_t)AI_NETWORK_IN_1_SIZE_BYTES; i++) {
        g_input[i] = (int8_t)((i * 37U) & 0x7FU) - 64;
    }
    in[0].data = AI_HANDLE_PTR(g_input);
    out[0].data = AI_HANDLE_PTR(g_output);

    /* One warm-up run first: the first call touches cold Flash and
     * populates the caches, and is not what steady-state inference costs. */
    g_results.stage = 4U;
    (void)ai_network_run(net, in, out);
    g_results.stage = 5U;

    for (i = 0; i < N_RUNS; i++) {
        t0 = DWT_CYCCNT;
        (void)ai_network_run(net, in, out);
        t1 = DWT_CYCCNT;
        g_results.cycles[i] = t1 - t0;
    }

    g_results.ai_error = 0;
    g_results.stage = 6U;                /* 6 = complete */
    for (;;) { }
}
