# Coref
wget "https://drive.google.com/file/d/1ScVz_o4V3G7watezLriCC0vU5gT7FO7q/view"
tar tar -xzvf wl_coref_transmucores.tar.gz -C ./coref_model 

# RE
wget "https://drive.usercontent.google.com/u/0/uc?id\u003d1UqOUdeK96m6EabI-cg2EeBz6p3IwrPZ6\u0026export\u003ddownload"
tar tar -xzvf files_indie.tar.gz -C ./RE_model

# EL
wget "https://dl.fbaipublicfiles.com/GENRE/fairseq_multilingual_entity_disambiguation.tar.gz"
tar tar -xzvf fairseq_multilingual_entity_disambiguation.tar.gz -C ./EL_model
wget -P ./EL_model "http://dl.fbaipublicfiles.com/GENRE/titles_lang_all105_marisa_trie_with_redirect.pkl"

