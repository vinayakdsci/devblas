use std::ptr;
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
    fn new(dvector: &'data mut [T], stride: usize, rows: usize, cols: usize) -> Self {
        Self {
            data: dvector,
            stride: stride,
            rows: rows,
            cols: cols,
        }
    }
    
    fn get_as_mut_ptr(&mut self) -> *mut T {
        self.data.as_mut_ptr()
    }
    
    fn get_as_ptr(&self) -> *const T {
        self.data.as_ptr()
    }
    
    fn data(&self) -> &[T] {
        self.data
    }
}

impl<T> Index<usize> for Matrix<'_, T> where T: NumericType {
    type Output = T;
    fn index(&self, index: usize) -> &Self::Output {
        let ptr = self.get_as_ptr();
        let t: *const T;
        
        if index >= self.data.len() {
            panic!("OOB access on matrix array not allowed!");
        }

        unsafe {
            t = ptr.add(index);
            &*t
        }
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    #[should_panic(expected="OOB access on matrix array not allowed!")]
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
        
        assert_eq!(mat[2], 3);
    }
    
    #[test]
    pub fn mutate_data_ptr() {
        let mut data = vec![1, 2, 3, 4];
        let mut mat = Matrix::new(&mut data, 1, 2, 2);
        
        unsafe {
            let ptr = mat.get_as_mut_ptr();
            ptr::write(ptr.offset(1), -1);
        }
        
        assert_eq!(mat[1], -1);
    }
}