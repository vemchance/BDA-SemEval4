# BDA word embeddings experiments

> Word embeddings experiments to determine which word embedding system to use

A key problem in this task is that of being tested on unseen languages we haven't trained on. The solution to this is not obvious, hence experimentation is needed.

This part of the project is headed up by Starbeamrainbowlabs [@sbrl](https://github.com/sbrl) <https://starbeamrainbowlabs.com/>.

As I discovered in [my hull science festival demo](https://starbeamrainbowlabs.com/blog/article.php?article=posts/533-research-smflooding-vis.html), different languages are often embedded to different spaces. The question then becomes how can we embed the meaning of words into the **same** space *regardless* of the source language. Doing so has the potential to increase our score significantly.

If this is not possible, then we'll hafta translate everything before it hits the model as a backup option.

Alternatively, we can guess the source language and train the model on data from all languages, but this is challenging unless we have some idea what the source language is which isn't possible in this instance.

## Getting started





## Notes

### Methodology summary
The dataset [Wikipedia-based Image Text Dataset (WIT)](https://github.com/google-research-datasets/wit/blob/main/DATA.md) was downloaded, and a wordlist extracted. Only unique words were kept with their order in the original dataset preserved. Wordlists were extracted for the following languages:

```
en de fr es ru it nl pl
```

Generated wordlists were then embedded with the following:

- **[CLIP](https://github.com/openai/CLIP):** ViT-B/32
- **[GloVe](https://nlp.stanford.edu/projects/glove/):** glove.twitter.27B.200d.txt

Invalid Unicode was ignored. Overlength words were trimmed to appropriate lengths where required, and any further overlength exceptions were ignored.



### Handling dataset
> [Wikipedia-based Image Text Dataset (WIT)](https://github.com/google-research-datasets/wit/blob/main/DATA.md)

...the file format is *horrible* but annoyingly compliant to its own standard. [mlr](https://miller.readthedocs.io/) seems capable of parsing it though. Like why don't you just provide JSONL?!?!

Convert to JSON Lines:

```bash
mlr --icsv --ifs "\t" --ojsonl cat wit_v1.train.all-1percent_sample.tsv.gz >somefile.jsonl
```


Extract count per-language:

```bash
zcat wit_v1.train.all-1percent_sample.jsonl.gz | jq '.language' | sort | uniq -c | sort -nr
```

From 1% sample file, languages with 10K+ samples: en de fr es ru it nl pl ja


Extract a deduplicated wordlist from descriptions:

```bash
zcat wit_v1.train.all-1percent_sample.jsonl.gz | jq --arg LANG "en" --raw-output 'select(.language == $LANG) | "\(.context_page_description) \(.context_section_description)"' | tr ',."()[]{}:;@#' ' ' | awk '{gsub(/[“”]|'"'"'\b|\b'"'"'/, " ", $0); gsub(/\s+/, "\n", $0); print tolower($0)}' | awk '!seen[$0]++' | gzip >some_filepath.txt.gz
```

The above, wrapped to run for every language in the above list:

```bash
for LANG in en de fr es ru it nl pl ja; do {(zcat wit_v1.train.all-1percent_sample.jsonl.gz | jq --arg LANG "$LANG" --raw-output 'select(.language == $LANG) | "\(.context_page_description) \(.context_section_description)"' | tr ',."()[]{}:;@#' ' ' | awk '{gsub(/[“”]|'"'"'\b|\b'"'"'/, " ", $0); gsub(/\s+/, "\n", $0); print tolower($0)}' | awk '!seen[$0]++' | gzip >"wordlist-$LANG.txt.gz"; echo "[ $(date) ] >>> LANG COMPLETE: $LANG" >&2;) &}; done
```

Post-process wordlists from `make_wordlists.sh`:

```bash
for LANG in en de fr es ru it nl pl ja; do echo ">>> $LANG"; fd -g "wordlist-${LANG}*.txt.gz" -0 | xargs -0 cat | awk '!seen[$0]++' | gzip --best >"wordlist-$LANG.txt.gz"; done
# https://crates.io/crates/fd-find / sudo apt install fd-find
fd -t f 'wordlist-[a-z]{2}_(train|test|val)-[0-9]+\.txt\.gz' -0 | xargs -0 rm

# Word count
fd -t f -g '*.txt.gz' | while read -r filepath; do echo -e "${filepath}\t$(zcat "${filepath}" | wc -l)"; done
```

Trim all wordlists down to the same size:

```bash
fd -t f -g '*.txt.gz' | while read -r filename; do {(zcat "${filename}" | head -n 95263 | gzip --best >"${filename%.*}-clipped.txt.gz") &}; done; wait
```

### Embeddings
Each file that embeds a wordlist with the target algorithm follows a standard pattern. Each one is named `embed_ALGORITHM.py`, where `ALGORITHM` is the name of the aforementioned algorithm.

The following environment variables are used:

- **`INPUT`:** Defines a filepath to the input wordlist to embed. Wordlist will be gzip compressed.
- **`OUTPUT`:** Defines a filepath to output embeddings to as [JSONL](https://jsonlines.org/).

All scripts use `lib/handle_open.py`, so should support gzipped (.gz) input and output transparently based on filename.

The following scripts exist:

- **`embed_clip.py`:** Embeds using [CLIP](https://github.com/openai/CLIP)
- **`embed_glove.py`:** Embeds using [GloVe](https://nlp.stanford.edu/projects/glove/). Requires the extra environment variable `GLOVE` to be set to a GloVe file (e.g. `glove.twitter.27B.200d.txt`).



## Known issues
- It is unlikely that Japanese is being tokenised correctly, given we simply look for any whitespace character and split on that. Ref <https://github.com/taishi-i/toiro>