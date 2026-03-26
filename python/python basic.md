# **快捷键**

## VScode

- **切换光标形状:** `Insert` 或 `Fn+Insert`。
- **返回上一个光标位置:** `Alt + ←` 
- **按单词移动:** `Ctrl + Left/Right Arrow` (
- **多行/多光标操作:** 按下 `Alt` 键并点击鼠标，或使用 `Ctrl+Alt+↑/↓`。
- **跳转行首/行尾:** `Home` 或 `End`

- `alt+F12`在变量上查看
- `ctrl`点击快速查询
- `ctrl+shift+p`打开命令控制
  - `Reload Window`刷新

# python

## 注释：鼠标悬停显示docstring

(显示的变量类型与程序的运行无关)

### 对变量注释

```python
from typing import List
train_files: List[str] = []
"""All training image paths"""
```

### Protocol

- 想对下面的`config.run_time`鼠标悬停显示注释

  ```python
  import ml_collections
  
  config = ml_collections.ConfigDict()
  config.run_name = "name"
  '''run name for wandb logging and checkpoint saving'''
  ```

​	`ml_collections.ConfigDict()` `class`无法显示docstring，`ConfigDict` 成员不是静态声明的字段，Pylance 无法知道有哪些 key :

- `protocol`增加docstring

  ```python
  from typing import Protocol,Any
  import ml_collections
  
  config = ml_collections.ConfigDict()
  config.run_name = "name"
  
  class config_hover(Protocol):
      run_name: float | Any
      """run name for wandb logging and checkpoint saving"""
  cast(config_hover,config)  # 我保证 value 在运行时就是 Type，请按这个类型对待它
  ```

### TypedDict

[TypedDict](TypedDict.md)类型是dict

```python
from typing import TypedDict

class TrainConfig(TypedDict):
    lr: float
    """Learning rate"""

    batch_size: int
    """Batch size per GPU"""

    epochs: int
    """Total training epochs"""

    weight_decay: float
    """AdamW weight decay"""

    data_root: str
    """Dataset root directory"""

```

### dataclass

```python
from dataclasses import dataclass,field
from typing import List

@dataclass  # 类没有方法，只有属性变量
class Sample:
    reward: List=[]  #错误
    reward: list = field(default_factory=list)  # 正确
    """Reward value of this sample. Higher is better."""

sample = Sample(reward=[0.91,0.83])
sample.reward # IDE 悬停 sample.reward 会显示字段 docstring
```

- 可变默认值必须用`default_factory`

### 类型推断红色波浪线

只要一个变量可能取不符合的值，就会推断出现红色

### 函数签名

函数传入的变量的key，类型，函数返回值（类似C++中函数声明）

### 类签名

```python
class zip(
    iter1: Iterable[Any],
    iter2: Iterable[Any],
    iter3: Iterable[Any],
    iter4: Iterable[Any],
    iter5: Iterable[Any],
    iter6: Iterable[Any],
    /,
    *iterables: Iterable[Any],
    strict: bool = False
)
```

这个 `/` 表示：

> **这 6 个参数是 \*只能位置传参\*（positional-only）**

也就是说你不能这样写：

```
zip(iter1=a, iter2=b)  # ❌ 非法
```

只能：

```
zip(a, b, c)
```

## decorator

```python
@fun
class MyClass:
    ...

# 等价于
class MyClass:
    ...
MyClass = fun(MyClass)
```

## import导入包

`python -m myproject.main` 用module 模式启动 Python

### package

在 Python 3.3+：

👉 即使 `ddpo/` 里 *没有* `__init__.py`，
 👉 你也可以 `import ddpo`，而且是“合法包”。
 这叫：Namespace Package（命名空间包）

如果当前目录下没有，ddpo文件夹在内层文件夹，能找到吗:找不到。除非它所在的“父目录”在 `sys.path` 里

## 传入参数

*强制后面的参数只能用“关键字参数”方式（`allow_dotted_keys=True`）传递，不能用位置参数

```python
class ConfigDict(
    initial_dictionary: Mapping[str, Any] | None = None,
    type_safe: bool = True,
    convert_dict: bool = True,
    *,
    allow_dotted_keys: bool = False,
    sort_keys: bool = True
)
```

### `**dict`

`**dict` 是什么？

- `**`：拆 **映射**（dict）

```
func(**kwargs)
```

等价于：

```
func(key1=d["key1"], key2=d["key2"], ...)
```

⚠️ 前提：

- dict 的 **key 必须是字符串**
- key 必须是 **合法的参数名**

### *args

`*args` = 把多个**位置参数**，打包成一个 tuple

# 读取和输出

### 读取输入

`input()`读取一行输入，应该是字符串

```python
N = int(input())
data = [list(map(int, input().split())) for _ in range(N)]
```

### 中位数，平均值

```python
import statistics
data=[[],[]] # shape(N,d) N个数据，每个数据d维度
median = statistics.median(row[1] for row in data)
```

# 其他

### `absl`

```python
from absl import app 

app.run(main)
# 等价
if __name__ == "__main__":
    main(sys.argv)
```

```py
from absl import flags
FLAGS = flags.FLAGS

flags.DEFINE_float("lr", 1e-4, "learning rate")
flags.DEFINE_integer("batch_size", 8, "Batch size")

def main(argv):
    print(FLAGS.lr, FLAGS.batch_size)

if __name__ == "__main__":
    app.run(main)
```

### `re`

re = Regular Expression,它是 Python 里专门用来： **查找 / 切分 / 匹配 / 替换文本模式的工具箱。**

 Regular Expression是描述字符串形状的公式，`^1\d{10}$`意思是：1 开头后面 10 个数字,匹配中国手机号 

```py
re.split(",|\.|\?|!|\"|:|;|\ ", "RichHF-18K: a dataset for rich human feedback on generative images.")
```



# python语法

x[0:4]：包含x[0]，不包含x[4]