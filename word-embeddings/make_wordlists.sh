#!/usr/bin/env bash

filepath_urls="$1";

if [[ -z "${filepath_urls}" ]]; then
	echo "Usage:
	./make_wordlists.sh {{urls_file.tsv}}

urls_file.tsv must be formatted with the first column as the postfix, and the second column as the url to download. Example:

example-0	https://example.com/example-0.tsv.gz
example-1	https://example.com/example-1.tsv.gz

NOTE: It is assumed that all input files are gzipped!
" >&2;
	exit 0;
fi

## Parse a .jsonl file into a series of wordlists.
# $1	filepath_input	The input JSONL file to parse - e.g. 
# $2	target_postfix	A postfix to append to the wordlist filenames. Output filepaths are in the form wordlist-LANG_POSTFIX.txt.gz and are saved to the current directory.
process_jsonl() {
	local filepath_input="$1"; # eg wit_v1.train.all-1percent_sample.jsonl.gz
	local target_postfix="$2";
	
	for LANG in en de fr es ru it nl pl ja; do
		{(zcat "${filepath_input}" \
			| jq --arg LANG "${LANG}" --raw-output 'select(.language == $LANG) | "\(.context_page_description) \(.context_section_description)"' \
			| tr ',."()[]{}:;@#' ' ' \
			| awk '{gsub(/[“”]|'"'"'\b|\b'"'"'/, " ", $0); gsub(/\s+/, "\n", $0); print tolower($0)}' \
			| awk '!seen[$0]++' \
			| gzip >"wordlist-${LANG}_${target_postfix}.txt.gz";
		echo "[ $(date) ] >>> LANG COMPLETE: ${LANG}" >&2;) &};
	done
	
	wait
}

user_agent="WordlistBot/1.0 ($(uname -srm | sed 's/ /; /g'); BDA-Hull-SemEval2024; Managed by @sbrl +https://starbeamrainbowlabs.com/) curl; Bash/${BASH_VERSION}";

while read -r postfix url; do
	tmpfile="srcfile-${postfix}.jsonl.gz";
	
	echo "[ $(date) ] >>> GET+CONVERT ${tmpfile} ← ${url}";
	
	curl -A "${user_agent}" "${url}" | gzip -dc | mlr --icsv --ifs "\t" --ojsonl cat | gzip >"$tmpfile";
	
	echo "[ $(date) ] >>> PROCESS ${tmpfile}";
	process_jsonl "${tmpfile}" "$postfix";
	
	echo "[ $(date) ] >>> DELETE ${tmpfile}";
	rm "${tmpfile}";
done <"${filepath_urls}";