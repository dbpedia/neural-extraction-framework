from chunking.chunking_model import *
from chunking.crf_chunker import *
from utils import *



class ChunckProcessor:
	use_gpu = False

	hyper_params = {
		'run_ID': 26,
		'createData': True,
		'bs': 8,
		'bert_model_name': 'xlm-roberta-base',
		'available_models': "'ai4bharat/indic-bert' 'bert-base-multilingual-cased' 'xlm-roberta-base' 'bert-base-multilingual-uncased' 'xlm-mlm-100-1280' 'xlm-mlm-tlm-xnli15-1024' 'xlm-mlm-xnli15-1024'",
		'alpha' : 0.00001,
		'epochs': 3,
		'rseed': 123,
		'nw': 4,
		'train_ratio' :  0.7,
		'val_ratio' : 0.1,
		'max_len' : 275,
		'which_way' : 3,
		'which_way_gloss': "1= take first wordpiece token id | 2= take last wordpiece token id | 3= average out the wordpiece embeddings during the forward pass",
		'embedding_way' : 'last_hidden_state',
		'embedding_way_gloss': 'last_hidden_state, first_two, last_two',
		'notation' : 'BI',
		'platform': 1,
		'available_platforms': "MIDAS server = 1, colab = 2",
		'chunker':'XLM' # CRF or XLM
	}

	model_embeddings = {
		'ai4bharat/indic-bert':768,
		'bert-base-multilingual-cased':768,
		'xlm-roberta-base':768,
		'bert-base-multilingual-uncased':768,
		'xlm-mlm-100-1280':1280,
		'xlm-mlm-tlm-xnli15-1024':1024,
		'xlm-mlm-xnli15-1024':1024
	}

	hyper_params['embedding_size'] = model_embeddings[hyper_params['bert_model_name']]

	my_tagset = torch.load('chunking/data/my_tagset_'+hyper_params['notation']+'.bin')

	hyper_params['my_tagset'] = my_tagset

	os.environ['PYTHONHASHSEED'] = str(hyper_params['rseed'])
	# Torch RNG
	torch.manual_seed(hyper_params['rseed'])
	torch.cuda.manual_seed(hyper_params['rseed'])
	torch.cuda.manual_seed_all(hyper_params['rseed'])
	# Python RNG
	np.random.seed(hyper_params['rseed'])
	random.seed(hyper_params['rseed'])

	is_cuda = torch.cuda.is_available()

	if is_cuda and use_gpu:
		device = torch.device("cuda:0")
		t = torch.cuda.get_device_properties(device).total_memory
		c = torch.cuda.memory_reserved(device)
		a = torch.cuda.memory_allocated(device)
		f = t -(c-a)  # free inside cache
		print("GPU is available", torch.cuda.get_device_name(), round(t/(1024*1024*1024)), "GB")
	else:
		device = torch.device("cpu")
		print("GPU not available, CPU used")

	hyper_params['device'] = str(device)

	if hyper_params['chunker'] == 'XLM':
		print('Creating the XLM chunker model...')
		model = chunker_class(device, hyper_params).to(device)
		checkpoint = torch.load('chunking/state_dicts/model/'+str(hyper_params['run_ID'])+'_epoch_4.pth.tar',map_location=device)
		
		# Handle version compatibility - ignore unexpected keys
		state_dict = checkpoint['state_dict']
		# Remove problematic keys that exist in newer transformers versions
		keys_to_remove = [k for k in state_dict.keys() if 'position_ids' in k]
		for key in keys_to_remove:
			print(f"Removing incompatible key: {key}")
			del state_dict[key]
		
		# Load with strict=False to handle any remaining mismatches
		missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
		if missing_keys:
			print(f"Missing keys: {missing_keys}")
		if unexpected_keys:
			print(f"Unexpected keys: {unexpected_keys}")
		print("Model loaded successfully!")
		
		tokenizer = AutoTokenizer.from_pretrained(hyper_params['bert_model_name'])
	elif hyper_params['chunker'] == 'CRF':
		model, tokenizer = 'CRF', 'CRF'

	def __init__(self, language):
		self.language = language
		self.nlp = load_stanza_model(self.language, self.use_gpu)

	def run(self, sentence):
		start_time = time.time()
		all_sents, exts, ctime, etime = perform_extraction(sentence,
													 self.language,
													 self.model,
													 self.tokenizer, 
													 self.nlp, show=False)
		ttaken = round(time.time() - start_time, 3)
		return all_sents, exts, ctime, etime, ttaken


if __name__=="__main__":
	sentence = "आदिम युग में सब लोग दिन भर काम से थक जाने के बाद मनोरंजन के लिए कही खुले में एक घेरा बनाकर बैठ जाते थे और उस घेरे के बीचों-बीच ही उनका भोजन पकता रहता , खान-पान होता और वही बाद में नाचना-गाना होता ।"
	processor = ChunckProcessor()
	data = processor.run(sentence)
	print(data)

