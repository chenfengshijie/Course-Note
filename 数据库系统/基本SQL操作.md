[toc]

# 关系演算

1. 选择
2. 投影
3. 自然连接
4. 笛卡尔乘积
5. 并
6. 外连接。左外连接、右外连接和全外连接。

# SQL语言简介

## 基本类型

1. char
2. varchar
3. int
4. smallint
5. numeric(p,d)定点数，
6. real，double precision。浮点数和双精度浮点数
7. float(n)。精度为n为的浮点数

```sql
create table r
    (dept_name varchar(20),
    building varchar(15),
    budget numeric(12,2)
    primary key(dept_name))
```

基本的完整性约束：

- 主键约束（Primary Key Constraint）：用于确保表中每行数据的唯一性，通常是一个或多个列的组合。
- 外键约束（Foreign Key Constraint）：用于确保两个表之间的数据一致性。外键约束要求一个表中的列必须在另一个表中存在，通常用于实现关系数据库中的关系。
- 唯一约束（Unique Constraint）：用于确保表中某一列或列组合的唯一性，不允许重复值出现。
- 检查约束（Check Constraint）：用于确保表中的数据符合指定的条件，例如限制某一列的值只能是特定的范围。
- 非空约束（Not Null Constraint）：用于确保表中某一列的值不能为空

## SQL查询

```sql
select ...
from ...
where condition
```

select指定需要的属性名，from中多个关系名通过笛卡尔乘积形式组成一个新的关系，where通过条件来限制from中生成的关系。

### 自然连接

```sql
select ...
from r1 natural join r2
where
```

SQL 自然连接是一种特殊的等值连接，它要求两个表中进行比较的属性组必须是名称相同的属性组，并且在结果中把重复的属性列去掉。

假设有两个表，一个是学生表，有学号和姓名两个列，另一个是成绩表，有学号和分数两个列。如果我们想查询每个学生的姓名和分数，我们可以使用自然连接，它会根据相同的列名（学号）来组合这两个表，并且去掉重复的列（学号）。自然连接的SQL语句如下：

```sql
SELECT student.name, score.grade
FROM student NATURAL JOIN score
ORDER BY score ASC
```

这样就可以得到每个学生的姓名和分数了。同时可以添加ORDER BY按照顺序排列。

### 集合运算

union、intersect、except来进行集合的运算。
好的，我来给你一个SQL集合运算的代码。SQL集合运算是指对两个或多个查询结果集进行的运算，它们有三种类型：并集（UNION），交集（INTERSECT）和差集（EXCEPT）。并集是把两个结果集合并起来，并且去掉重复的行；交集是把两个结果集中都存在的行选出来；差集是把第一个结果集中存在而第二个结果集中不存在的行选出来²⁴。

例如，假设有两个表，一个是学生表，有学号和姓名两个列，另一个是课程表，有课程号和课程名两个列。如果我们想查询所有学生的姓名和所有课程的名称，我们可以使用并集运算：

```sql
SELECT name FROM student
UNION
SELECT cname FROM course;
```

如果我们想查询既是学生又是老师的人的姓名，我们可以使用交集运算：

```sql
SELECT name FROM student
INTERSECT
SELECT name FROM teacher;
```

如果我们想查询只是学生而不是老师的人的姓名，我们可以使用差集运算：

```sql
SELECT name FROM student
EXCEPT
SELECT name FROM teacher;
```

### 聚集函数

SQL聚集函数是对一组数据进行汇总的函数，输入是一组数据的集合，输出是单个值¹²。SQL提供了五个常用的聚集函数，分别是：

- COUNT：计算某列或表中非空值的个数
- SUM：计算某列中数值型数据的总和
- AVG：计算某列中数值型数据的平均值
- MAX：找出某列中最大的值
- MIN：找出某列中最小的值

例如，假设有一个表叫student，有三个列分别是sno（学号），name（姓名）和score（成绩）。如果我们想查询学生表中有多少个学生，我们可以使用COUNT函数：

```sql
SELECT COUNT(*) FROM student;
```

如果我们想查询学生表中所有学生的成绩总和，我们可以使用SUM函数：

```sql
SELECT SUM(score) FROM student;
```

如果我们想查询学生表中所有学生的成绩平均值，我们可以使用AVG函数：

```sql
SELECT AVG(score) FROM student;
```

如果我们想查询学生表中成绩最高的学生的姓名和成绩，我们可以使用MAX函数和子查询：

```sql
SELECT name, score FROM student
WHERE score = (SELECT MAX(score) FROM student);
```

如果我们想查询学生表中成绩最低的学生的姓名和成绩，我们可以使用MIN函数和子查询：

```sql
SELECT name, score FROM student
WHERE score = (SELECT MIN(score) FROM student);
```

### 分组

好的，我来给你一个SQL分组的详细信息和代码。SQL分组是指使用GROUP BY子句按照某个字段或多个字段中的值，将相同的数据分为一组¹²。SQL分组通常与聚集函数一起使用，以对每个分组进行汇总统计³。

例如，假设有一个表叫employee，有四个列分别是eid（员工号），ename（姓名），dept（部门）和salary（工资）。如果我们想查询每个部门的员工数，我们可以使用GROUP BY子句和COUNT函数：

```sql
SELECT dept, COUNT(*) FROM employee
GROUP BY dept;
```

如果我们想查询每个部门的平均工资，我们可以使用GROUP BY子句和AVG函数：

```sql
SELECT dept, AVG(salary) FROM employee
GROUP BY dept;
```

如果我们想查询每个部门中工资最高的员工的姓名和工资，我们可以使用GROUP BY子句、MAX函数和子查询：

```sql
SELECT ename, salary FROM employee
WHERE (dept, salary) IN (
  SELECT dept, MAX(salary) FROM employee
  GROUP BY dept);
```

同时可以支持使用having子句对分组进行筛选。

## SQL的添加、删除、修改

SQL添加、删除、修改是指使用INSERT、DELETE和UPDATE语句对数据表中的数据进行增加、删除和修改，或者使用ALTER TABLE语句对数据表中的列进行添加、删除和修改 。

例如，假设有一个表叫employee，有四个列分别是eid（员工号），ename（姓名），dept（部门）和salary（工资）。如果我们想在表中添加一条新的记录，我们可以使用INSERT语句：

```sql
INSERT INTO employee VALUES (1001, '张三', '开发部', 8000);
```

如果我们想从表中删除一条记录，我们可以使用DELETE语句：

```sql
DELETE FROM employee WHERE eid = 1001;
```

如果我们想修改表中的一条记录，我们可以使用UPDATE语句：

```sql
UPDATE employee SET salary = 9000 WHERE eid = 1001;
```

如果我们想在表中添加一个新的列，比如email（邮箱），我们可以使用ALTER TABLE语句：

```sql
ALTER TABLE employee ADD email VARCHAR(50);
```

如果我们想从表中删除一个列，比如email（邮箱），我们可以使用ALTER TABLE语句：

```sql
ALTER TABLE employee DROP COLUMN email;
```

如果我们想修改表中的一个列的数据类型或长度，比如把salary（工资）从INT类型改为DECIMAL(10,2)类型，我们可以使用ALTER TABLE语句：

```sql
ALTER TABLE employee MODIFY salary DECIMAL(10,2);
```


### SQL权限操作

SQL权限操作是指使用GRANT、REVOKE和DENY语句对数据库中的用户或角色授予或撤销某些数据对象（如表、视图、存储过程等）的访问权限。

例如，假设有一个数据库叫mydb，有一个表叫employee，有四个列分别是eid（员工号），ename（姓名），dept（部门）和salary（工资）。如果我们想给一个用户叫user1授予employee表的查询和插入权限，我们可以使用GRANT语句：

```sql
GRANT SELECT, INSERT ON mydb.employee TO user1;
```

如果我们想撤销user1对employee表的插入权限，我们可以使用REVOKE语句：

```sql
REVOKE INSERT ON mydb.employee FROM user1;
```

如果我们想禁止user1对employee表的删除权限，我们可以使用DENY语句：

```sql
DENY DELETE ON mydb.employee TO user1;
```



