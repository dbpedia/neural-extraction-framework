# Coref
gdown 1ScVz_o4V3G7watezLriCC0vU5gT7FO7q -O wl_coref_transmucores.tar.gz 
tar -xzvf wl_coref_transmucores.tar.gz -C ./coref_model

# RE
gdown 1UqOUdeK96m6EabI-cg2EeBz6p3IwrPZ6 -O files_indie.tar.gz
tar -xzvf files_indie.tar.gz -C ./RE_model

# EL
wget "https://dl.fbaipublicfiles.com/GENRE/fairseq_multilingual_entity_disambiguation.tar.gz"
tar -xzvf fairseq_multilingual_entity_disambiguation.tar.gz -C ./EL_model
wget -P ./EL_model "http://dl.fbaipublicfiles.com/GENRE/titles_lang_all105_marisa_trie_with_redirect.pkl"

# remove the tar.gz files after extraction (run if successful)
# rm wl_coref_transmucores.tar.gz files_indie.tar.gz fairseq_multilingual_entity_disambiguation.tar.gz