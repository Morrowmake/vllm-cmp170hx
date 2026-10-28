# Community recipes

A community recipe is a way to serve a model on this fork that someone outside
Morrowmake made and maintains: a model we don't support, a different card
count or layout, or different settings. Good ones are listed in the
[README](README.md#community-recipes) so others can find them.

Community recipes are made and maintained by their authors; Morrowmake has not
tested them.

## How to submit

### 1. Make your recipe

1. Put the recipe in your own public repository.
2. Add a licence file to that repository.
3. Pin an exact fork version, a release tag or a full 40-character commit hash, never a branch.
   Example: `VLLM_COMMIT=18305948ac...` (all 40 characters) or `tag qwen38-recipe-1.0.0`.
4. If you use our container image, pin it by digest.
   Example: `image@sha256:<64-hex digest>`
5. State the hardware: card model, card count, and anything unusual about the host.
6. State the layout and settings: parallelism (for example `TP=2` or `PP=4`), context length, and every non-default flag or environment variable.
7. Publish your own measured numbers and exactly how you measured them (command, prompt and output lengths, concurrency).
8. Name yourself as the contact for issues and say where to report them (usually your repo's issues).
9. Make it start with one command if you can. See our
   [GLM-5.3-Flash](https://github.com/Morrowmake/glm53-flash-cmp170hx-recipe) and
   [Qwen3.8-Flash-Next](https://github.com/Morrowmake/qwen38-flash-next-cmp170hx-recipe)
   recipes for the shape we mean.
10. Don't say or suggest that Morrowmake made, tested, endorses or supports it.

### 2. Open a pull request

1. Fork this repository.
2. Add one row to the [Community recipes](README.md#community-recipes) table in `README.md`, in this format:
   `| [name](url) | Model | Hardware/layout | Fork version | @author |`
   Example: `| [qwen38-2x-long-context](https://github.com/you/qwen38-2x-long-context) | Qwen3.8-Flash-Next | 2× CMP 170HX, TP=2 | qwen38-recipe-1.0.0 | @you |`
3. If the table still shows the "None yet" line, remove it.
4. Open a pull request titled `Community recipe: <name>`.
5. Paste the checklist below into the description and tick each box.

```markdown
- [ ] The recipe is in my own public repository: <url>
- [ ] The repository has a licence
- [ ] It pins an exact fork version (release tag or full commit hash): <pin>
- [ ] If it uses the Morrowmake image, it pins the image by digest: <digest or "not used">
- [ ] It states hardware, layout and settings
- [ ] It includes my own measured numbers and how I measured them
- [ ] It names me as the contact for issues
- [ ] It does not claim Morrowmake made, tested, endorses or supports it
- [ ] My PR adds exactly one row to the README table, in the given format
```

### 3. What happens next

- We check three things: the recipe pins an exact fork version (and image digest if it uses our image), it makes no claim of our endorsement, and it has a licence.
- We don't run or test the recipe, and we don't check your numbers.
- We review pull requests as time allows and reply on the PR, either merging it or saying what needs to change.
- A listing may be removed if the recipe stops working, its repository goes away, or it is no longer maintained. Open a new PR once it's fixed and it can come back.

## Support

We don't support community recipes and can't answer questions about them;
please take those to the recipe's author.

## Credit

Authors keep their names and their repositories. A listing links to your work
and credits you; nothing is moved or copied here.
