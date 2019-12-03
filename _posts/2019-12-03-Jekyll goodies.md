---
published: false
layout: post
title: Jekyll tips and tricks
permalink: /jekyll-tips/
---

## What is allowed to run on GitHub

Here are the GitHub approved Jekyll plugins:
https://pages.github.com/versions/

You can also check https://pages.github.com/versions.json, for more programmable way.


This [next list](https://jekyllcodex.org/without-plugins/) is plugin free solution.

For example this script:

    {% capture words %}
    {{ content | number_of_words | minus: 180 }}
    {% endcapture %}
    {% unless words contains '-' %}
    {{ words | plus: 180 | divided_by: 180 | append: ' minutes to read' }}
    {% endunless %}

Provides the read time based on number of words.


## Removing non printable characters from Unicode.

Often we don't know that text contains *byte order mark* or BOM character. This is used to indicate Unicode encoding of a text file. 

BOM used `\ufeff` unicode value and is optional so we can remove it. 

There are few way as I checked online:


### Simple ascii

Unicode string may have `\u200c` character that we may move to ascii.

    fp = open("file.txt", errors="replace", encoding="utf8")
    s = fp.read()
    a = s.decode("ascii", "ignore")
    s = a.encode("utf-8")

The problem with this; we remove all characters that are non ascii.


### Codec

First `utf-8-sig` codec:

    fp = open("file.txt", errors="ignore", encoding="utf8")
    s = fp.read()
    u = s.decode("utf-8-sig")

That gives you a unicode string without the BOM. You can then use

    s = u.encode("utf-8")

to revert to Unicode `utf-8` representation in this case. 






There are hundreds of control characters in unicode. If you are sanitizing data from the web or some other source that might contain non-ascii characters, you will need Python's unicodedata module. The unicodedata.category(…) function returns the unicode category code (e.g., control character, whitespace, letter, etc.) of any character. For control characters, the category always starts with "C".

This snippet removes all control characters from a string.

import unicodedata
def remove_control_characters(s):
    return "".join(ch for ch in s if unicodedata.category(ch)[0]!="C")

Examples of unicode categories:

>>> from unicodedata import category
>>> category('\r')      # carriage return --> Cc : control character
'Cc'
>>> category('\0')      # null character ---> Cc : control character
'Cc'
>>> category('\t')      # tab --------------> Cc : control character
'Cc'
>>> category(' ')       # space ------------> Zs : separator, space
'Zs'
>>> category(u'\u200A') # hair space -------> Zs : separator, space
'Zs'
>>> category(u'\u200b') # zero width space -> Cf : control character, formatting
'Cf'
>>> category('A')       # letter "A" -------> Lu : letter, uppercase
'Lu'
>>> category(u'\u4e21') # 両 ---------------> Lo : letter, other
'Lo'
>>> category(',')       # comma  -----------> Po : punctuation
'Po'
>>>



import string
filtered_string = filter(lambda x: x in string.printable, myStr)
