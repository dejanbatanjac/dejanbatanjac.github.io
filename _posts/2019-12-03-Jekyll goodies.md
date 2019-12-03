---
published: false
layout: post
title: Jekyll tips and tricks
permalink: /jekyll-tips/
---

## What is allowed to run on GitHub

Here are the [GitHub approved Jekyll plugins](https://pages.github.com/versions/), also [available in json](https://pages.github.com/versions.json).
But you may use Liquid without any plugins. The [next list](https://jekyllcodex.org/without-plugins/) is plugin free.
Here are all Jekyll [variables](https://jekyllrb.com/docs/variables/).
Here are all GitHub supported [themes](https://pages.github.com/themes/), and there are also the possible [errors](https://help.github.com/en/github/working-with-github-pages/troubleshooting-jekyll-build-errors-for-github-pages-sites#file-is-a-symlink).


## Removing non printable characters from Unicode.

From the [docs](https://jekyllrb.com/docs/installation/windows/) :

>If you use UTF-8 encoding, make sure that no BOM header characters exist in your files or very, very bad things will happen to Jekyll. This is especially relevant when you’re running Jekyll on Windows.

Often we don't know that text contains *byte order mark* or BOM character that is used to indicate Unicode encoding of a text file. 
There are few ways to remove BOM unicode character.

### Simple ascii way of removing unicode characters

Unicode string may have BOM (`\ufeff`) or other non printable (`\u200c`)characters that we may move to ascii first.

    fp = open("file.txt", errors="replace", encoding="utf8")
    s = fp.read()
    a = s.decode("ascii", "ignore")
    s = a.encode("utf-8")

The problem with this, we remove all characters that are non ascii.


### Codec utf-8-sig

We may use `utf-8-sig` codec:

    fp = open("file.txt", errors="ignore", encoding="utf8")
    s = fp.read()
    u = s.decode("utf-8-sig")

That gives you a unicode string without the BOM. Later you can revert it to unicode:

    s = u.encode("utf-8")


### Removing by unicode category

There are hundreds of control characters in unicode. This Python snippet removes all control characters from a string.

    import unicodedata
    def remove_control_characters(s):
        return "".join(ch for ch in s if unicodedata.category(ch)[0]!="C")

Examples of unicode categories:

    from unicodedata import category
    category('\r')      # carriage return --> Cc : control character
    'Cc'
    category('\0')      # null character ---> Cc : control character
    'Cc'
    category('\t')      # tab --------------> Cc : control character
    'Cc'
    category(' ')       # space ------------> Zs : separator, space
    'Zs'
    category(u'\u200A') # hair space -------> Zs : separator, space
    'Zs'
    category(u'\u200b') # zero width space -> Cf : control character, formatting
    'Cf'
    category('A')       # letter "A" -------> Lu : letter, uppercase
    'Lu'
    category(u'\u4e21') # 両 ---------------> Lo : letter, other
    'Lo'
    category(',')       # comma  -----------> Po : punctuation
    'Po'

One handy trick may be to filter the printable characters:

    import string
    filtered_string = filter(lambda x: x in string.printable, string)

## Jekyll pages

Jelyll *pages* are standalone content (not date based). Very simple website should have `index.md` page, `about.md` page and `contact.md` page.

The `.md` extension will be converted later to the `.html` extension and moved into `_site` folder.

However, when you call the page you may not write the `.html` extension in the end.

    .
    |-- about.md    # => http://example.com/about (or about.html)
    |-- index.md    # => http://example.com/
    └── contact.md  # => http://example.com/contact

If you use the permalink structure, the URL changes. For example if the `about.md` page has permalink set:

    ---
    permalink: /about-me/
    ---

then the output link will be the `http://example.com/about-me.html`.

## Jekyll posts

When working with posts we often need to write the list of all posts and URLs.

    <ul>
    {% for post in site.posts %}
        <li>
        <a href="{{ post.url }}">{{ post.title }}</a>
        </li>
    {% endfor %}
    </ul>

Sometimes we need that list but with the except:

    <ul>
    {% for post in site.posts %}
        <li>
        <a href="{{ post.url }}">{{ post.title }}</a>
        {{ post.excerpt }}
        </li>
    {% endfor %}
    </ul>

## Jekyll tags and categories

Very often we need to list of categories:

    <ul>
    {% for category in site.categories %}
    <li>{{ category[0] }}</li>
    {% endfor %}
    </ul>  

Similar for all tags:

    <ul>
    {% for tag in site.tags %}
    <li>{{ tag[0] }}</li>
    {% endfor %}
    </ul> 


## Jekyll list all posts based on categories

    {% for category in site.categories %}
    <h3>{{ category[0] }}</h3>
    <ul>
        {% for post in category[1] %}
        <li><a href="{{ post.url }}">{{ post.title }}</a></li>
        {% endfor %}
    </ul>
    {% endfor %}

As you note we used just Liquid for all previous examples. No Ruby code.
[Liquid](https://shopify.github.io/liquid/) is templating language to process templates.

In Liquid you output content like this:

    {{ variable }} 
    
You perform if like this:

    {% if statement %}

Liquid also used [filters](https://jekyllrb.com/docs/liquid/filters/) and [tags](https://jekyllrb.com/docs/liquid/tags/).

