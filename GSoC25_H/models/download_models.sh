# Coref
gdown 1ScVz_o4V3G7watezLriCC0vU5gT7FO7q -O wl_coref_transmucores.tar.gz 
tar -xzvf wl_coref_transmucores.tar.gz -C ./coref_model
# rm wl_coref_transmucores.tar.gz # remove the tar.gz file after extraction

# RE
gdown 1UqOUdeK96m6EabI-cg2EeBz6p3IwrPZ6 -O files_indie.tar.gz
tar -xzvf files_indie.tar.gz -C ./RE_model
# rm files_indie.tar.gz # remove the tar.gz file after extraction

# EL
wget "https://dl.fbaipublicfiles.com/GENRE/fairseq_multilingual_entity_disambiguation.tar.gz"
tar -xzvf fairseq_multilingual_entity_disambiguation.tar.gz -C ./EL_model
wget -P ./EL_model "http://dl.fbaipublicfiles.com/GENRE/titles_lang_all105_marisa_trie_with_redirect.pkl"

# rm fairseq_multilingual_entity_disambiguation.tar.gz # remove the tar.gz file after extraction
