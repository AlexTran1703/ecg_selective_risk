/* STM32F411xC/E: 512 KB Flash, 128 KB SRAM.
 *
 * The last 512 bytes of SRAM are carved out for the result block at a
 * fixed address, so the host can read timings over SWD without needing
 * the ELF symbol table.
 */

ENTRY(Reset_Handler)

MEMORY
{
  FLASH   (rx)  : ORIGIN = 0x08000000, LENGTH = 512K
  RAM     (rwx) : ORIGIN = 0x20000000, LENGTH = 127K
  RESULTS (rw)  : ORIGIN = 0x2001FE00, LENGTH = 512
}

_estack = ORIGIN(RAM) + LENGTH(RAM);

SECTIONS
{
  .isr_vector : { KEEP(*(.isr_vector)) } > FLASH

  .text :
  {
    *(.text*)
    *(.rodata*)
    . = ALIGN(4);
  } > FLASH

  .ARM.exidx : { *(.ARM.exidx*) } > FLASH
  _sidata = LOADADDR(.data);

  .data :
  {
    . = ALIGN(4);
    _sdata = .;
    *(.data*)
    . = ALIGN(4);
    _edata = .;
  } > RAM AT > FLASH

  .bss :
  {
    . = ALIGN(4);
    _sbss = .;
    *(.bss*)
    *(COMMON)
    . = ALIGN(4);
    _ebss = .;
  } > RAM

  .results (NOLOAD) : { KEEP(*(.results)) } > RESULTS

  /DISCARD/ : { *(.ARM.attributes) }
}
