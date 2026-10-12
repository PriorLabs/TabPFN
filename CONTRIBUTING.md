# Contributing to TabPFN

Thanks for helping improve TabPFN. Bug reports, fixes, docs and use cases are all welcome. This page covers what we expect from contributions and how we handle them.

## Pull requests

- You can open a pull request directly; you don't need to open an issue first. For larger features or API changes, open an issue or start a discussion first so we can agree on the direction before you invest time.
- Fill in the pull request template, including a changelog fragment (see [`changelog/README.md`](changelog/README.md)).
- Keep each PR focused on one change, with tests that fail before the change and pass after it.
- External contributors can have at most **3 open pull requests** at a time. We may close additional ones without review; reopen them once earlier ones are merged or closed.
- The first time you open a pull request, you'll be asked to sign our Contributor License Agreement (CLA).

## What we fix

TabPFN has many parameters, and not every combination is supported. We prioritise problems on the default path and in documented usage (the docs, README and `examples/`).

For undocumented combinations of settings, we usually prefer a clear error that points to a supported alternative over adding code paths and tests to make the combination work. If you need such a combination for a real use case, open an issue describing the use case and we'll consider supporting it.

## AI-assisted contributions

You may use AI tools to write code, tests, docs or issues. We follow the same principle as projects such as [LLVM](https://llvm.org/docs/AIToolPolicy.html) and [SciPy](https://docs.scipy.org/doc/scipy/dev/conduct/ai_policy.html): the tools are fine, but a human stays accountable for every contribution.

1. **A human owns it.** You have reviewed every change, understand it and can explain it in your own words. You're responsible for its correctness, licensing and tests, as with code you typed yourself.
2. **Disclose it.** Tick the AI-assistance box in the pull request or issue template.
3. **Speak for yourself.** Respond to review comments personally. Don't paste in AI-generated replies to reviewer questions. Using AI to translate or fix grammar is fine. If an AI tool wrote your pull request description, start it with a short summary of the changes in your own words.
4. **No fully automated submissions.** Agents must not open pull requests or issues without a human reviewing each one before it is submitted. Automation set up by the maintainers, such as Dependabot, is exempt.
5. **Report real problems.** Issues should describe a problem you or a user actually ran into, or one you have reproduced with the minimal example in the bug template. Bulk-filed reports from automated bug-hunting are not welcome.

## When the guidelines aren't followed

Maintainers may close any pull request or issue that doesn't follow this page, without a detailed review, with a link to this page. If a contributor keeps breaking these guidelines after that, we may block them from the repository. A block needs two maintainers to agree.

If you think we closed something by mistake, reply on the pull request or issue and a maintainer will take another look.

This isn't personal. Maintainer time is limited, and these rules let us spend it on contributions where a person is engaged.

Maintainers can close with this message:

> Thanks for the contribution. We're closing this because it doesn't follow our [contribution guidelines](https://github.com/PriorLabs/TabPFN/blob/main/CONTRIBUTING.md). If you think we got this wrong, reply here and a maintainer will take another look.
