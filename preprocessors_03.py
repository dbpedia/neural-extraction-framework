# +

import logging 
from typing import Optional

from snorkel.preprocess import preprocessor
from snorkel.types import DataPoint

# # -
# logger = logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG)

# # formatter = logging.Formatter('')
# file_handler = logging.FileHandler('test_extractData.log')
# logger.addHandler(file_handler)


@preprocessor()
def get_NPs_text(cand: DataPoint) -> DataPoint:
    """
    Returns the text for the two mentions in candidate
    """
    cand.NPs = cand['pairs']
    
    return cand




@preprocessor()
def get_text_between(cand: DataPoint) -> DataPoint:
    """
    Returns the text between the two NPs mentions in the sentence
    """
    start = cand.ele1_word_idx[1] + 1
    end = cand.ele2_word_idx[0]
    cand.text_between = cand.tokens[start:end]
    
#     cand.text_between = " ".join(cand.tokens[start:end])
#     logger.debug(" ".join(cand.tokens[start:end]))
#     logger.debug(cand.text_between)
   
    return cand


@preprocessor()
def get_left_tokens(cand: DataPoint) -> DataPoint:
    """
    Returns tokens in the length 2 window to the left of the NPs mentions
    """
    # TODO: need to pass window as input params
    window = 2

    end = cand.ele1_word_idx[0]+1
    cand.ele1_left_tokens = cand.tokens[0:end][-1 - window : -1]

    end = cand.ele2_word_idx[0]+1
    cand.ele2_left_tokens = cand.tokens[0:end][-1 - window : -1]
    
#     logger.debug("tokens: {}".format(cand.tokens))
#     logger.debug("ele1_left_token: {}".format(cand.ele1_left_tokens))
#     logger.debug("ele2_left_token: {}".format(cand.ele2_left_tokens))
    
    return cand



@preprocessor()
def get_right_tokens(cand: DataPoint) -> DataPoint:
    """
    Returns tokens in the length 2 window to the right of the NPs mentions
    """
    # TODO: need to pass window as input params
    window = 2

    end = cand.ele1_word_idx[0]+1
    cand.ele1_right_tokens = cand.tokens[end : end+ window]

    end = cand.ele2_word_idx[0]+1
    cand.ele2_right_tokens = cand.tokens[end : end+ window]
    
#     logger.debug("tokens: {}".format(cand.tokens))
#     logger.debug("ele1_right_token: {}".format(cand.ele1_right_tokens))
#     logger.debug("ele2_right_token: {}".format(cand.ele2_right_tokens))
    
    
    return cand
