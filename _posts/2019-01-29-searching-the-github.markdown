---
layout: post
title: Searching the GitHub
date: '2019-01-29 10:26:30 +0100'
categories: search
published: true
---
There are couple of documents [here](https://help.github.com/articles/understanding-the-search-syntax/), [here](https://help.github.com/articles/searching-code/) and [here](https://help.github.com/articles/about-searching-on-github/) that explain [GitHub advanced search](https://github.com/search/advanced).

### In GitHub you search inside:
* Repositories
* Topics
* Issues and pull requests
* Code
* Commits
* Users
* Wikis

### Search is not case sensitive
* `cat` is the same as `CAT`

### Based on time created or pushed 
* `cat created:>2016-04-29` or 
* `cat pushed:>2016-04-29`

### Language search
* `cat language:javascript` (matches repositories with the word "cat" written in JavaScript)
* `cat -language:javascript` (matches repositories with the word "cat" not written in JavaScript)

### Topics or label search
* `cat topics:>=5` (matches repositories with the word "cat" 5 or more topics)
* `label:"in progress"` 

### Filter `in`
* `cat in:file` (matches code where "cat" appears in the file contents)
* `cat in:path` (matches code where "cat" appears in the file path)
* `cat in:file,path` (matches code where "cat" appears in the file contents or the file path) 


### User search
* `user:dejanbatanjac extension:txt` (matches code from @dejanbatanjac that ends in `.txt`)

### Sarching from certain organization
* `org:github extension:js` (matches code from GitHub that ends in `.js`)

### Searching based on stars
* `cat stars:>1000` (will search for repos that have `>1000` stars)

### File size
* `cats size:<10000` (match files smaller than `10000` Bytes)

### Filename search
* `filename:cat path:test language:ruby` (matches Ruby files named `cat` within the `test` directory)

### Extension search
* `icon size:>200000 extension:css` (matches files larger than `200 KB` that end in `.css` and have the word `icon`)
 
### Date
* `cat created:>2018-12-31` (search files created starting from 2019)

### NOT
* `hello NOT world` (matches repositories that have the word `hello` but not the word `world`)

### Search in [forks](https://help.github.com/articles/searching-in-forks/)
* `fork:true` (matches all repositories containing the word "github," including forks)

### [Sorted](https://help.github.com/articles/sorting-search-results/) search
* `cat sort:updated` (matches repositories containing the word `cat`sorted by most recently updated date)
