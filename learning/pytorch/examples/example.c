/**
To compile: gcc -o a.out example.c
 */

#include <stdio.h>
#include "iacaMarks.h"

int kernel2(int max) {
  int acc = 1;
  for (int i = 0; i < max; i++) {
    IACA_START
      // C CODE GOES HERE
      acc += 1;
  }
  IACA_END
  return acc ;
}

int kernel(int max) {
  IACA_START
  int acc = 1;
  acc += 1;
  acc += 2;
  acc += 3;
  int acc2 = acc;
  kernel2(5);
  IACA_END
  return acc ;
}

int main(int argc, char **argv) {
  printf("kernel: %d\n", kernel(10));
  IACA_START
  IACA_END
  return 0;
}
