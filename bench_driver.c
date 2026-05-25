#include <devblas/devblas.h>
#include <assert.h>
#include <stddef.h>

int main(void) {
	devblas_sgemm(DEVBLAS_LAYOUT_ROW_MAJOR, false, false, 0, 0, 0, NULL,
				  0, NULL, 0, NULL, 0);

	return 0;
}
