#include <asm/hwcap.h>
#include <stdio.h>
#include <sys/auxv.h>

int main() {
  unsigned long hwcaps = getauxval(AT_HWCAP);
  if (hwcaps & HWCAP_ASIMDHP)
    printf("FP16 NEON (ASIMDHP) supported\n");
  else
    printf("FP16 NEON not supported\n");
  return 0;
}