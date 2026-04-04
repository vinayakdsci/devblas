use std::ops::Index;

/// Type constraints for Matrix.
pub trait NumericType : Copy {}
impl NumericType for i32 {}
impl NumericType for i64 {}
impl NumericType for f32 {}
impl NumericType for f64 {}

/// A generic Matrix view defined for i32/64 and f32/64.
#[derive(Debug)]
pub struct Matrix<'data, T> {
    data: &'data mut [T],
    stride: usize,
    rows: usize,
    cols:  usize,
}

impl<'data, T> Matrix<'data, T> where T: NumericType {
    pub fn new(dvector: &'data mut [T], stride: usize, rows: usize, cols: usize) -> Self {
        Self {
            data: dvector,
            stride: stride,
            rows: rows,
            cols: cols,
        }
    }
    
    pub fn data(&self) -> &[T] {
        self.data
    }
    
    pub fn data_mut(&mut self) -> &mut [T] {
        self.data.as_mut()
    }
    
    pub unsafe fn get_unchecked_mut(&mut self, index: usize) -> &mut T {
        unsafe {
            self.data.get_unchecked_mut(index)
        }
    }

    pub unsafe fn get_unchecked(&self, index: usize) -> &T {
        unsafe {
            self.data.get_unchecked(index)
        }
    }
    
}

/// Should NOT be USED in the hot loop.
impl<T> Index<usize> for Matrix<'_, T> where T: NumericType {
    type Output = T;
    fn index(&self, index: usize) -> &Self::Output {
        &self.data[index]
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    #[should_panic]
    pub fn panic_oob() {
        let mut data = vec![1, 2, 3, 4];
        let mat = Matrix::new(&mut data, 1, 2, 2);
        
        // PANIC!
        let _t = mat[500];
    }

    #[test]
    pub fn create_matrix_and_index() {
        let mut data = vec![1, 2, 3, 4];
        let mat = Matrix::new(&mut data, 1, 2, 2);
        
        unsafe {
            assert_eq!(*mat.get_unchecked(2), 3);
        }
    }
    
    #[test]
    pub fn mutate_data_ptr() {
        let mut data = vec![1, 2, 3, 4];
        let mut mat = Matrix::new(&mut data, 1, 2, 2);
        
        unsafe {
            *mat.get_unchecked_mut(1) = -1;
        }
        
        assert_eq!(mat[1], -1);
    }
}