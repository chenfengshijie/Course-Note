# 1. 教学管理系统

$$
\begin{align}
  

1. \quad & \Pi_{S,GRADE}(\sigma_{C='001'}(S\Join SC)\cap \sigma_{C='003'}(S\Join SC))
\\
2. \quad & \Pi_{S,SNAME,GRADE}(\sigma_{C='001'}(S\Join SC))
\\
3.\quad & \Pi_{SNAME,AGE}(S\Join SC - \sigma_{C='001'}(S\Join SC))
\\
4. \quad & \Pi_{SNAME}(\sigma_{TEACHER='gao' \cup GRADE >= 90}(S\Join SC \Join C))
\\
5. \quad & \Pi_{SNAME,C}(S \Join SC) \div \Pi_{C}(C)
\end{align}
$$

# 2. 工程管理数据库

$$
\begin{aligned}
1. \quad & \Pi_{P}(\sigma_{SCITY='北京'\cap COLOR='蓝色'\cap SNAME='S1'}(SPJ\Join S\Join P\Join J))
\\
2. \quad & \Pi_{P,JNAME}(\sigma_{SCITY=JCITY}(SPJ\Join S \Join J))
\\
3. \quad & \Pi_{P}(P)-\Pi_{P}(\sigma_{JCITY='长春'}(SPJ\Join p \Join j))
\\
4. \quad & \Pi_{J,JNAME}(\sigma_{P='P2'}(SPJ\Join P \Join J))
\\
5. \quad & \Pi_{S,SNAME}(\sigma_{J='J5'\cap COLOR='绿色'}(SPJ\Join S \Join P \Join J))
\end{aligned}
$$

# 3. 关系代数验证参照完整性约束

$$
\Pi_{F}(S) - \Pi_{K}(R)
$$

# 5. 企业管理数据库

```sql
 1) 一号部门（D# = 1）员工的个数
SELECT COUNT(*) FROM Employee NATURAL JOIN Department WHERE D = 1;
# 2) 查询每个部门的部门 ID 和员工数量
SELECT D,COUNT(E) FROM Employee NATURAL JOIN Department GROUP BY D;
# 3) 查询“技术部”员工工资超过 10000 的员工姓名
SELECT Employee.NAME FROM Employee NATURAL JOIN Department WHERE D='技术部' AND salary > 10000;
# 4) 查询所有部门的平均工资，返回部门 ID 和平均工资（avgSalary）
SELECT D,AVG(salary) FROM Employee NATURAL JOIN Department GROUP BY D;
# 5) “技术部”中姓张的员工的个数
SELECT COUNT(*) FROM Employee WHERE E IN (
    SELECT E FROM Employee WHERE D IN (
        SELECT D FROM Department WHERE Dname='技术部'
    ) AND NAME LIKE '张%');
```

# 6. 图书管理系统

```sql

#1) 查询借阅了超过 5 本书的学生学号
SELECT Sno FROM Borrow 
GROUP BY Sno HAVING COUNT(B#) > 5;
#2) 查询借阅了“人民教育出版社”出版的书籍的学生姓名和年龄，按年龄降#序排列
SELECT Sname,Sage FROM Student WHERE Sno IN(
    SELECT Sno FROM Borrow NATURAL JOIN BOOK
    WHERE Publisher = "人民教育出版社"
);
#3) 查询借阅的所有图书的借阅时长都超过 90 天的学生学号
SELECT Sno FROM Student WHERE NOT EXISTS(
    SELECT * FROM Borrow 
    WHERE Sno = Student.Sno AND Time <= 90
);
#4) 查询书名包含“Big%Date”的图书书名和对应的数量
SELECT Title, COUNT(*) AS Count
FROM Book
WHERE Title LIKE 'Big%Date'
GROUP BY Title;
#5) 查询超过 5 名“CS”系的不同学生借阅的书的书名
SELECT DISTINCT Book.Title
FROM Book NATURAL JOIN Borrow NATURAL JOIN Student 
WHERE Student.Sdept = 'CS'
GROUP BY Book.Title
HAVING COUNT(DISTINCT Student.Sno#) > 5;

```

# 7. 借阅图书

![1681133296020](image/homework/1681133296020.png)

# 8.课程管理

![1681134755012](image/homework/1681134755012.png)

# 9.工程管理

![1681136820804](image/homework/1681136820804.png)