---
layout: post
title: "辗转相除法"
categories: algorithms
tags: algorithms

---

可以使用辗转相除法求解两个数的最大公约数。

## Java 实现

### 递归版

```java
class GCD {
    public int gcd(int a, int b) {
        if (b == 0) {
            return a;
        } else {
            return gcd(b, a % b);
        }
    }
```

### 迭代版

```java
class GCD {
    public int gcd(int a, int b) {
        while (b != 0) {
            int tmp = a % b;
            a = b;
            b = tmp;
        }
        return a;
    }
}
```

## References

1. [LeetCode 2807 力扣官方题解](https://leetcode.cn/problems/insert-greatest-common-divisors-in-linked-list/solutions/2589529/zai-lian-biao-zhong-cha-ru-zui-da-gong-y-udrs/)
