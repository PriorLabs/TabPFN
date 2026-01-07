# Changelog Fragments

This directory contains changelog "fragments" - small files that describe changes in each PR.

## How to Add a Changelog Entry

1. Create a file named `<PR_NUMBER>.<category>.md` in this directory
2. Write a short, user-friendly description of your change

### Example

For PR #712 that adds a new feature:

```bash
# Using towncrier create (recommended - validates the category)
towncrier create 712.added.md --content "Add support for custom loss functions in finetuning"

# Or manually
echo "Add support for custom loss functions in finetuning" > changelog/712.added.md
```

## Categories

| Category | Filename | Description |
|----------|----------|-------------|
| Breaking | `<PR>.breaking.md` | Breaking changes (requires user action) |
| Added | `<PR>.added.md` | New features |
| Changed | `<PR>.changed.md` | Changes to existing functionality |
| Fixed | `<PR>.fixed.md` | Bug fixes |
| Deprecated | `<PR>.deprecated.md` | Deprecated features |

## Multiple Entries per PR

You can have multiple changelog entries for one PR. Use different types as needed:

```bash
712.added.md    # New feature
712.fixed.md    # Bug fix in same PR
```

If you need **multiple entries of the same type**, add a counter:

```bash
712.added.1.md  # First new feature
712.added.2.md  # Second new feature
```

Pattern: `<PR>.<type>.md` or `<PR>.<type>.<counter>.md`

## When is a Changelog Entry Required?

A changelog entry is required for all PRs unless a maintainer adds the **"no changelog needed"** label.

Examples where "no changelog needed" is appropriate:
- Documentation-only changes
- CI/tooling changes
- Test refactoring (not bug fixes)
- Typo fixes

## For Maintainers: Releasing

When releasing a new version, run:

```bash
towncrier build --version X.Y.Z
```

This will:
1. Compile all fragments into `CHANGELOG.md`
2. Delete the fragment files
3. Add a new `## [Unreleased]` section
