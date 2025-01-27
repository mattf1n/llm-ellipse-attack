for mode in 1 2
do
  tokenfile=data/mode${mode}_tokens.txt
  tokens=$(wc -l <$tokenfile)
  echo "For mode $mode ($tokens tokens)"
  tmpfile=$(mktemp)
  sed 's/^ //' $tokenfile > $tmpfile
  space_first_words=$(wc -l <$tmpfile)
  sfw_pct=$(echo "$space_first_words / $tokens" | bc -l)
  echo "$space_first_words space-first words ($sfw_pct percent)" 
  num_words=$(pv /usr/share/dict/words | fgrep -xcaFf $tmpfile)
  words_pct=$(echo "$space_first_words / $tokens" | bc -l)
  echo "$num_words word tokens ($words_pct percent)"
  tmpfile=$(mktemp)
done
