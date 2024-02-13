---
layout: post
title: "Java 里的 Comparator 和 Comparable"
categories: java
tags: java
math: true

---

## Comparator 接口

### 实现 Comparator 接口

一般使用场景为用 `Collections.sort()` 或 `Arrays.sort()` 进行自定义排序。需要实现 `compare(T first, T second)` 方法：

```java
public interface Comparator<T> {
	int compare(T first, int second);
}
```

- 写成匿名类的形式：

    ```java
    Collections.sort(collection_of_T, new Comparator<T> {
    {
        @Override
        public int compare(T first, T second) {
            // do something;
        }
    }});
    ```

- 使用 Lambda 表达式（Java 8）：

    ```java
    Collections.sort(collection_of_T, (o1, o2) -> do something);
    ```

### Comparator 的静态方法

Comparator 接口提供了很多便捷的方法来创建比较器。使用《Java 核心技术卷一》的例子，假设对一个 Person 数组进行排序。

```java
interface Person {
	String getFirstName();
	String getMiddleName();
	String getLastName();
	default String getName() {
		return "";
	}
}
```

- 按姓名进行排序：

    ```java
    Arrays.sort(people, 
      Comparator.comparing(Person::getName));
    ```

- 先按姓排序，再按名排序：

    ```java
    Arrays.sort(people, 
      Comparator.comparing(Person::getLastName).thenComparing(Person::getFirstName));
    ```

- 按照姓名长度排序：

    ```java
    Arrays.sort(people, 
      Comparator.comparing(Person::getName, 
      (s, t) -> Integer.compare(s.length(), t.length())));
    ```

    更简便的写法：

    ```java
    Arrays.sort(people, 
    Comparator.comparingInt(p -> p.getName().length()));
    ```

- 按照中间名排序，使用 `nullsFirst` 或 `nullsLast` 处理 `null` ：

    ```java
    import java.util.Comparator.*;

    Arrays.sort(people, comparing(Person::getMiddleName, nullsFirst(naturalOrder()))); // 自然顺序
    Arrays.sort(people, comparing(Person::getMiddleName, nullsFirst(reverseOrder()))); // 逆序
    ```

## Comparable 接口

实现 Comparable 接口使得一个对象能与同类型的对象进行比较。

```java
public class MyClass implements Comparable<MyClass> {
    @Override
    public int compareTo(MyClass other) {
		// do something
    }
}
```

## References

1. [Java 8 Lambda: Comparator](https://stackoverflow.com/questions/44225896/java-8-lambda-comparator)
2. Cay S. Horstmann. Java 核心技术卷一（原书第 12 版）. 机械工业出版社, 2022.
