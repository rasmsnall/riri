include!("interp.rs");
fn main() {
    println!("sum_to_ten = {}", sum_to_ten());
    println!("signed_division = {}", signed_division());
    println!("truncating_cast = {}", truncating_cast());
    println!("sign_extending_cast = {}", sign_extending_cast());
    println!("comparisons = {}", comparisons());
    println!("shifts = {}", shifts());
    let overflowed = std::panic::catch_unwind(overflows).is_err();
    println!("overflows panicked = {overflowed}");
}
