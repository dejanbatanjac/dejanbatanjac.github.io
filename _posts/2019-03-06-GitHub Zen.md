---
published: true
---
This little program will output random GitHub zen rules (in fact these are the zen rules of Python) `import this`:

~~~
import requests
def zen():
	'''
	Calls GitHub zen
	'''
	url = 'https://api.github.com/zen'
	page = requests.get(url)
	print(page.content)
    
zen()
~~~

I will try to comment on some.

### Practicality beats purity
If this means you forget about linter, please do.
Being practical, means to sacrifice purity.
No one asks for perfect code in first commit.

### Speak like a human.
Explain your code, so that other humans can understand.
Including you after 2 months.

### Keep it logically awesome.
You just need to have clear idea what you do.

### Avoid administrative distraction.
Don't let anyone stop you from coding.

### Half measures are as bad as nothing at all.
Just keep in mind, you are not finished until you tested everything you can think of.

### Responsive is better than fast.
Create programs that can talk.

### Approachable is better than simple.
Still, it's good to be simple.

### Design for failure.
Write `try` blocks and exceptions.

### Non-blocking is better than blocking.
Write async services.

### Favor focus over features.
Just essential features at the very start. 

### It's not fully shipped until it's fast.
Setup must be easy as 1-2-3. In case of Python it should run from Github (setyp.py)
