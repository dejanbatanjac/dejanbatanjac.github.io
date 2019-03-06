---
published: false
---
This little program will output what GitHub code zen rules:

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

## Practicality beats purity
Follow the rules of being practical.
If this means you forget about linter, or best practices, please do.
Also don't even try to write perfect code from once. Commit early.

## Speak like a human.
Explain your code, so that other humans can understand.
Including you after 2 months.

## Keep it logically awesome.
It doesn't say you need to create perfect code.
You just need to have clear idea what you do.

## Avoid administrative distraction.
Don't let anyone stop you from coding.

## Half measures are as bad as nothing at all.
Just keep in mind, you are not finished until you tested everything you can think of.

## Responsive is better than fast.
Create programs that can talk.

## Approachable is better than simple.
Still, it's good to be simple.

## Design for failure.
Write `try` blocks.

## Non-blocking is better than blocking.
Write async services.

## Favor focus over features.
Just essential features at the very start. 

## It's not fully shipped until it's fast.
Setup must be easy as 1-2-3.
If it's not you are not done.






