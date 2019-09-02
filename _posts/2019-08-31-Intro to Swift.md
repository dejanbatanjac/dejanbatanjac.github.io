---
published: false
layout: post
title: Swift intro
---

I would compare new Swift programming language with Python/PyTorch especially in the field on Ai.

### What maters about Swift

Swift (Apple Swift) was built to be fast. Chris Lattner who created the LLVM compiler, created Swift in 2010 as an *ambitious* programming language.

Swift is low level programming language - means you can deal with hardware and write bootloaders in it.

Recently, and to compete with PyTorch, Swift added Automatic Differentiation engine from Fortran. We can use that engine just by setting `@differentiable` keyword on any function.

Here is the [full detailed guided tour](https://docs.swift.org/swift-book/GuidedTour/GuidedTour.html) to Swift.

### Calling C

Python has GIL (Global Interpreter Lock) problem. When Python code needs to call into C, it performs slow, and Swift is made to work with C-like languages by design.

Swift can deal with C header files thanks to the Clang engine part of [LLVM](https://llvm.org/).

Clang can deal with C/C++/Objective C and even CUDA. 

Long term, Swift is trying to subtract C/C++ code out of the picture, because Swift is the new C.

### Can we use Python scientific libraries from Swift

Yes. This is possible. Just import Python in Swift (`import Python`), and from there import all the libraries you use to work with such as Numpy, Matplotlib and like.

    public let np = Python.import('numpy)

### Meat *var* and *let*

If you are new to Swift you will first meat keywords define a variable `var` and constant `let`.

Both of these need initialization at the very start. What happens to be a good practice now is a must:

    let implicitInteger = 70
    let implicitDouble = 70.0
    let explicitDouble: Double = 70

Initialization lets the compiler infer variable type. 

Note `let` is like a constant and `var` can alter it's value:

    var myVariable = 42
    myVariable = 50
    let myConstant = 42 # cannot be altered

### Arrays and dict

    var arr = [1,2,3,4]
    var arr2 : [Int] = arr
    var arr3 : Array<Int> = arr

In previous lines, `[Int]` was just a surger for `Array<Int>`. 
You can always do this:

    arr[1] = 5

For the dictionary type you also use same `[]` (square brackets). 

    var dict = [
        "Malcolm": "Captain",
        "Kaylee": "Mechanic",
    ]
    dict["Jayne"] = "Public Relations"

> Same as in Python arrays and dicts grows with `append`.

    dict.append("Bruce": "Movie star")

Here is empty array and empty dict:

    arr = []
    dict = [:]


### Functions

Once you pass variables and constant the next Swifty thing you will dig into will be functions. In Swift you use the `func` keyword.

> Note: In Python you sed `def` keyword for the same.

So a simple Swift function would be

    func more42(x : Int) -> Int {
        return x+42
    }

This function called `more42` would take and return Integer value. You need to set clearly the input parameters name and type and the output parameter type.

So you will call the previous function like this:

    more42(x=10)

> Note that Swift uses `{}` (curly brackets) where Python uses indent.
You could also write this:

    func more42(_ x : Int) -> Int {
        return x+42
    }

This means you can call the previous function like this without setting the parameter name:

    more42(10)


If function needs to return a tuple it totally can. Here is how we can define this:

    func more42(x : Int) -> (Int,Int) {
        return x+42, x+43
    }


...