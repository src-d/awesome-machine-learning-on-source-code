# Contributing Guidelines

awesome-machine-learning-on-source-code project is [CC-BY-SA 4.0 licensed](https://creativecommons.org/licenses/by-sa/4.0/) and accepts contributions via GitHub pull requests and twits to [@srcd_](https://twitter.com/srcd_).
This document outlines some of the conventions on submission workflow, commit message formatting, contact points, and other resources to make it easier to get your contribution accepted.


## Support Channels

The official support channels, for both users and contributors, are:

- GitHub [issues](https://github.com/src-d/awesome-machine-learning-on-source-code/issues)*
- Slack: #machine-learning room in the [source{d} Slack](https://join.slack.com/t/sourced-community/shared_invite/enQtMjc4Njk5MzEyNzM2LTFjNzY4NjEwZGEwMzRiNTM4MzRlMzQ4MmIzZjkwZmZlM2NjODUxZmJjNDI1OTcxNDAyMmZlNmFjODZlNTg0YWM)

*Before opening a new issue or submitting a new pull request, it's helpful to
search the project - it's likely that another user has already reported the
issue you're facing, or it's a known issue that we're already aware of.


## How to Contribute

Pull Requests (PRs) are the main way to contribute.
In order for a PR to be accepted it needs to pass a list of requirements:

- Follow the link template*.
- Be correctly classified as a paper, a blog post, a talk, a software or a dataset.
- Not be a duplicate.
- All the PRs have to pass the personal evaluation of at least one of the [maintainers](MAINTAINERS.md).

*Link template:

```
* [Title](URL) - Description.
```

Description should not start with "A" or "The". For papers, it should include the authors.

### Format of the commit message

The commit summary must start with "Add" and end with a dot.
```

### Deprecation

A listed repository should be deprecated if:

* Repository's owner explicitly says that "this library is not maintained".
* Not committed for long time (2~3 years).

To mark a repository as deprecated, put `[DEPRECATED]` before the title and move it at the bottom of the list.

