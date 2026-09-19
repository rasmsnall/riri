/// A CUDA-style three-dimensional extent or index.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Dim3 {
    pub x: u32,
    pub y: u32,
    pub z: u32,
}

impl Dim3 {
    pub const fn new(x: u32, y: u32, z: u32) -> Self {
        Self { x, y, z }
    }

    /// Total number of elements covered by this extent.
    pub const fn count(&self) -> u32 {
        self.x * self.y * self.z
    }

    /// Converts a linear index (x fastest) into coordinates within `extent`.
    pub(crate) fn from_linear(i: u32, extent: Dim3) -> Dim3 {
        let x = i % extent.x;
        let y = (i / extent.x) % extent.y;
        let z = i / (extent.x * extent.y);
        Dim3 { x, y, z }
    }
}

impl From<u32> for Dim3 {
    fn from(x: u32) -> Self {
        Dim3::new(x, 1, 1)
    }
}

impl From<(u32, u32)> for Dim3 {
    fn from((x, y): (u32, u32)) -> Self {
        Dim3::new(x, y, 1)
    }
}

impl From<(u32, u32, u32)> for Dim3 {
    fn from((x, y, z): (u32, u32, u32)) -> Self {
        Dim3::new(x, y, z)
    }
}
