---
title: "Jekyll Github 블로그에 MathJax 적용하기"
date: 2020-03-09 00:16:00
categories: [Blogging, Setting]
tags: [Jekyll, MathJax]
use_math: true
seo:
  date_modified: 2020-03-09 03:16:48 +0900
---







 [**Jekyll Github 블로그에 MathJax로 수학식 표시하기**](https://mkkim85.github.io/blog-apply-mathjax-to-jekyll-and-github-pages/) 를 참고하여 적용했습니다.

<br/>

<br/>

✔️ 마크다운 엔진 변경

- `_config.yml` 파일의 내용을 아래와 같이 수정
- 아래 내용은 Jekyll의 default 설정에 해당됩니다. ([참조](https://jekyllrb.com/docs/configuration/default/))


```yml
# Conversion
markdown: kramdown
highlighter: rouge
lsi: false
excerpt_separator: "\n\n"
incremental: false
```

<br/>

<br/>

✔️ `mathjax_support.html` 파일 생성

- `_includes` 디렉토리에 `mathjax_support.html` 파일 생성 후 아래 내용 입력
- `MathJax`의 설정과 에러 메세지 출력을 포함하고 있습니다.

```html
<script type="text/x-mathjax-config">
MathJax.Hub.Config({
    TeX: {
      equationNumbers: {
        autoNumber: "AMS"
      }
    },
    tex2jax: {
    inlineMath: [ ['$', '$'] ],
    displayMath: [ ['$$', '$$'] ],
    processEscapes: true,
  }
});
MathJax.Hub.Register.MessageHook("Math Processing Error",function (message) {
	  alert("Math Processing Error: "+message[1]);
	});
MathJax.Hub.Register.MessageHook("TeX Jax - parse error",function (message) {
	  alert("Math Processing Error: "+message[1]);
	});
</script>
<script type="text/javascript" async
  src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-MML-AM_CHTML">
</script>
```
<br/>

<br/>

✔️ `_layouts/default.html` 파일의 `head.html` 부분에 아래 내용 삽입

- 아래 if 문으로 `use_math` 가 적용된다면, `mathjax_support.html` 을 포함합니다. 결국 `MathJax` 를 사용한다는 뜻이 됩니다.

{% raw %}
```html
/* mathjax 추가 */
{% if page.use_math %}
  {% include mathjax_support.html %}
{% endif %}
```
{% endraw %}

<br/>

<br/>

✔️ YAML front-matter 설정

- 수학식을 표시할 포스트의 front-matter에 `use_math: true` 적용

```markdown
---
title: "Jekyll Github 블로그에 MathJax 적용하기"
date: 2020-03-08 12:14:00 +0800
categories: [Blogging, Configuration]
tags: [Jekyll, MathJax]
use_math: true
---
```

<br/>

