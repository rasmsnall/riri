// Bodies for the interpreter to run. Each takes no arguments and returns a
// scalar, so the expected answer can be written down beside it.

pub fn sum_to_ten() -> u32 {
    let mut total = 0u32;
    let mut i = 0u32;
    while i < 10 {
        total += i;
        i += 1;
    }
    total
}

pub fn signed_division() -> i32 {
    let a = -7i32;
    let b = 3i32;
    a / b
}

pub fn truncating_cast() -> u8 {
    let wide = 300u32;
    wide as u8
}

pub fn sign_extending_cast() -> i64 {
    let narrow = -3i8;
    narrow as i64
}

pub fn comparisons() -> bool {
    let a = -1i32;
    let b = 1i32;
    a < b
}

pub fn shifts() -> u32 {
    let x = 1u32;
    x << 5
}

pub fn overflows() -> u8 {
    let mut x = 250u8;
    let mut i = 0u8;
    while i < 10 {
        x += 1;
        i += 1;
    }
    x
}
