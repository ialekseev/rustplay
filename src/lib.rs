

#[cfg(test)]
mod tests {
    use super::*;

    #[test]    
    fn test_vec() {
        
        // {binds immutably} to {immutable value}
        let a: Vec<i32> = vec![1, 2];        
        //a = vec![3, 4]; //error: cannot assign twice to immutable variable `a`
        
        // {binds mutably} to {possibly mutable value}
        let mut b: Vec<i32> = vec![1, 2];        
        b = vec![3, 4]; //OK
        assert_eq!(b, vec![3, 4]);
        
        // {binds immutably} to {a reference to} {immutable value}
        let c: &Vec<i32> = &vec![1, 2];        
        //c = &vec![3, 4]; //error: cannot assign twice to immutable variable `c`        
        //*c = vec![3, 4]; //error: cannot assign to `*c`, which is behind a `&` reference
        
        // {binds immutably} to {a reference to} {mutable value}
        let d: &mut Vec<i32> = &mut vec![1, 2];        
        //d = &mut vec![3, 4]; //error: cannot assign twice to immutable variable `d` 
        *d = vec![3, 4]; //OK    
        assert_eq!(d, &vec![3, 4]);

        // {binds mutably} to {a reference to} {immutable value}
        let mut e: &Vec<i32>= &vec![1, 2];        
        let new_e: Vec<i32> = vec![3, 4];
        e = &new_e; //OK
        assert_eq!(e, &vec![3, 4]);
        //*e = vec![3, 4]; //cannot assign to `*e`, which is behind a `&` reference

        // {binds mutably} to {a reference to} {mutable value}
        let mut f: &mut Vec<i32>= &mut vec![1, 2];        
        let mut new_f: Vec<i32>= vec![3, 4];        
        f = &mut new_f; //OK
        assert_eq!(f, &vec![3, 4]);
        *f = vec![3, 4]; //OK
        assert_eq!(f, &vec![3, 4]);        
    }

    #[test]    
    fn test_int() {
        
        // {binds immutably} to {immutable value}
        let a: i32 = 1;        
        //a = 2; //error: cannot assign twice to immutable variable `a`
        
        // {binds mutably} to {possibly mutable value}
        let mut b: i32 = 1;        
        b = 2; //OK
        assert_eq!(b, 2);
        
        // {binds immutably} to {a reference to} {immutable value}
        let c: &i32 = &1;        
        //c = &2; //error: cannot assign twice to immutable variable `c`        
        //*c = 2; //error: cannot assign to `*c`, which is behind a `&` reference
        
        // {binds immutably} to {a reference to} {mutable value}
        let d: &mut i32 = &mut 1;        
        //d = &mut 2; //error: cannot assign twice to immutable variable `d` 
        *d = 2; //OK    
        assert_eq!(d, &2);

        // {binds mutably} to {a reference to} {immutable value}
        let mut e: &i32= &1;                
        e = &2; //OK
        assert_eq!(e, &2);
        //*e = 3; //cannot assign to `*e`, which is behind a `&` reference

        // {binds mutably} to {a reference to} {mutable value}
        let mut f: &mut i32= &mut 1;        
        let new_f: &mut i32=  &mut 2;        
        f = new_f; //OK
        assert_eq!(f, &2);
        *f = 2; //OK
        assert_eq!(f, &2);        
    }
    
    #[test]  
    fn test_struct() {
        #[derive(PartialEq, Debug)]
        struct A { a: i32}
        
        let mut t: A = A {a:1};        
        t = A {a:2}; //OK
        assert_eq!(t, A{a:2});
        t.a = 3;
        assert_eq!(t, A{a:3});
    }
}
