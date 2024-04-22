---
layout: post
title: "辗转相除法：GCD 与 LCM"
categories: algorithms
tags: algorithms
math: true

---

力扣中的许多题都涉及求最大公约数 (Greatest Common Divisor, GCD) 和最小公倍数 (Least Common Multiple, LCM)。本文记录了 GCD 与 LCM 的代码实现。

## GCD

可以使用辗转相除法求解两个数的最大公约数。

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
}

// 一行搞定
class GCD {
    public int gcd(int a, int b) {
        return b == 0 ? a : gcd(b, a % b);
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

## LCM

两个整数的最小公倍数与最大公因数之间有如下的关系：

$$
LCM(a, b) = \frac{|a\cdot b|}{GCD(a, b)}
$$

因此可以基于 `GCD(int a, int b)` 函数实现 `LCM(int a, int b)` 函数。

```java
class LCM {
    public int lcm(int a, int b) {
        return a * b / gcd(a, b);
    }
}
```

## References

1. [LeetCode 2807 力扣官方题解](https://leetcode.cn/problems/insert-greatest-common-divisors-in-linked-list/solutions/2589529/zai-lian-biao-zhong-cha-ru-zui-da-gong-y-udrs/)
2. [LeetCode 3116 灵神题解](https://leetcode.cn/problems/kth-smallest-amount-with-single-denomination-combination/solutions/2739205/er-fen-da-an-rong-chi-yuan-li-pythonjava-v24i)
3. [最小公倍数](https://www.wikiwand.com/zh-hans/%E6%9C%80%E5%B0%8F%E5%85%AC%E5%80%8D%E6%95%B8)
