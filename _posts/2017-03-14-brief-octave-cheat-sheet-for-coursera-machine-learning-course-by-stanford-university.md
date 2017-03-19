---
title: Brief Octave Cheat Sheet for Coursera Machine Learning Course by Stanford University
layout: post
category: Machine Learning
---

I use R with Python a lot, Octave is the chosen language in Coursera course: Machine Learning by Stanford University.

So this article will only cover necessary concept to finish Machine Learning course.

# Index

In Octave, matrix and vector are indexed from 1, which differs from many other languages.

# Output

One line not ending with a semicolon will print the result to output, with semicolon with suppress that output.

Or use `disp(i);` or `sprintf(i)`

# String

## Compare
strcmp, strmatch

# Cell-array

## Acess

    ca = cell(2,1); % create cell array
    ca{1} = 'abc'; % assign to first element
    ca{2} = 'def'; % assign to second element
    ca{1}; % access first element

# Range

`1:10` will create a 1x10 matrix or say 10-element vector with numbers from 1 to 10. `1:2:10` will create vector with each other numbers from 1 to 10, i.e. `[1 3 5 7 9]`. The middle 2 is the specified step.

# Matrix, Vector

## Assignment

    % assign a matrix to A: 
    A = [1 2; 3 4; 5 6]
    A =
        1   2
        3   4
        5   6

    % assign to second column
    >> A(:,2) = [10 11 12]
    % space, comma or semicolon doesn't matter here
    >> A(:,2) = [10, 11, 12]
    >> A(:,2) = [10; 11; 12]

## Access

    % access A at first row and second column.
    >> A(1,2)
    ans = 2

    % access second row, here colon refers to all columns
    >> A(2,:)
    ans =
        3   4

    % access second column
    >> A(:,2)
    ans =
        2
        4
        6

    a(2)       # result is a scalar
    a(1:2)     # result is a row vector
    a([1; 2])  # result is a column vector

    a = [1, 2; 3, 4]
    all of the following expressions are equivalent and select the first row of the matrix.

    a(1, [1, 2])  # row 1, columns 1 and 2
    a(1, 1:2)     # row 1, columns in range 1-2
    a(1, :)       # row 1, all columns

    a(1:end/2)        # first half of a => [1, 2]
    a(end + 1) = 5;   # append element
    a(end) = [];      # delete element
    a(1:2:end)        # odd elements of a => [1, 3]
    a(2:2:end)        # even elements of a => [2, 4]
    a(end:-1:1)       # reversal of a => [4, 3, 2 , 1]

## Fill

    >> A = ones(2, 3) # ones or zeros
    A =
       1   1   1
       1   1   1

    >> rand(2, 3)
    ans =
       0.47210   0.10022   0.35182
       0.69316   0.71345   0.71179
       
## Concatenate

    >> B = [20 21; 22 23; 24 25]
    >> C = [A B] % concatenate A B horizontally
    ans =
        1   2   20   21
        3   4   22   23
        5   6   24   25

    >> D = [A; B] % concatenate A B vertically
    ans =
        1   2
        3   4
        5   6
        20  21
        22  23
        24  25

## Transpose

    >> A' % transpose
    ans =
        1   3   5
        2   4   6

## Max, Min
    >> max(magic(4)) % return every column's max
    ans =
        16  14  15  13

    >> max(magic(4), [], 2) % return every row's max
    ans =
       16
       11
       12
       15

    >> [val, ind] = max(magic(4)) % retuns every column's max and their index in corresponding column
    val =
        16  14  15  13
    ind =
        1   4   4   1

    >> max(max(A)); % max element in matrix
    >> A(:) % convert matrix into one column
    ans =
        1
        3
        5
        2
        4
        6
    >> max(A(:)) % max element in matrix
    ans =
        6

## Sum

    >> sum(A) % sum of column
    >> sum(A, 2) % sum of row

## Sum of diagonals in a square matrix

    >> M = magic(4);
    >> sum(sum(M.*eye(4))) % sum of diagonal top left to bottom right
    ans = 34
    >> sum(sum(M.* flipud(eye(4) ))) % sum of diagonal bottom left to top right
    ans = 34

## Flip

    >> flipud(A) $ flip matrix upside down
    ans =
        5   6
        3   4
        1   2

## Matrix select

    A(A==2)

## Reshape

    >> reshape(A, 2, 3)
    ans =
       1   5   4
       3   2   6

# Functions & control statements

Functions are saved in files with the file-ending .m for MATLAB. 

    function y = function_name(x1, ...
        x2) % x2 is optional
        if ~exist('x2', 'var') || isempty(x2)
            x2 = 1;
        end        
        y = x1 + x2;
    % y is the return value
    % x1 is a parameter
    % is also possible to return multiple values
    function [y1, y2] = function_name(x1)
        y1 = x1^2
        y2 = x1^3

    >> for i=1:10
    >>      disp(i)
    >> end;

    >> i = 1;
    >> while (i ~= 10)
    >>      disp(i);
    >>      i = i+1;
    >> endwhile;

    % i = 10
    >> if (i == 10)
    >>      sprintf('yes')
    >> else
    >>      sprintf('no')
    >> endif
    ans = yes

## Anonymous function

    @(x1, x2) another_func(x1, x2) % anonymous function, just like Python lambda

# Logic operations

not equal ~=

logical AND &&

logical OR ||

logical XOR xor(1,0)


**Reference**:

https://gist.github.com/obstschale/7320846

http://folk.ntnu.no/joern/itgk/refcard-a4.pdf