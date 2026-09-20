// A kernel-shaped function for the driver to look at.
pub fn scatter(out: &mut [u32], stride: usize, tid: usize, block: usize) {
    let i = block * stride + tid;
    if i < out.len() {
        out[i] = i as u32;
    }
}
