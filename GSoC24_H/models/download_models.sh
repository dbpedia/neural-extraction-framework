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

# Move files to their correct locations in the project structure:
# Move GSoC24_H/models/coref_model/model/xlmr_multi_plus_hi2.pt to GSoC24_H/models/coref_model/xlmr_multi_plus_hi2.pt
# Move GSoC24_H/models/RE_model/files_indie/my_tagset_BI.bin to GSoC24_H/input/my_tagset_BI.bin
# Move GSoC24_H/models/RE_model/files_indie/26_epoch_4.pth.tar to new folder tree GSoC24_H/models/state_dicts/model/26_epoch_4.pth.tar
# Move GSoC24_H/models/RE_model/files_indie/sklearn_crf_model_v2_pos_mapped_2.pkl to chunking/state_dicts/model/sklearn_crf_model_v2_pos_mapped_2.pkl
# Move GSoC24_H/models/EL_model/titles_lang_all105_marisa_trie_with_redirect.pkl to GSoC24_H/input/titles_lang_all105_marisa_trie_with_redirect.pkl