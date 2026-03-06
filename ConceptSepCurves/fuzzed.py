from typing import Collection, Sequence, Callable
import random
from itertools import product, islice


def random_article_insertion(sentence:Sequence[str], 
                             articles:tuple[str,...], 
                             variants:int=3) -> Collection[str]:
    """
    This method computes a collection of random article inserted sentences.
    The maximum number of sentences is equal to the length of the sentence * articles supplied.
    As such the variants returned will be from this collection.    
    
    Parameters
    ----------
    sentence : Sequence[str]
        The sentence in a split set of words.
    articles : tuple[str,...]
        The articles to insert at random locations
    variants : int, optional
        How many alternatives ought to be computed, if None is passed all options are used. The default is 3.

    Returns
    -------
    Collection[str]
        Sentences of words with a random selected article at random

    """
    
    # first create a collection locations with articles as tuples (this encompasses all possible solutions)
    insertion_options = list( 
        product([i for i in range(len(sentence))],
                [i for i in range(len(articles))])
        )
    # shuffle these locations
    random.shuffle(insertion_options)
    # select up to the requested number of options
    selected_options = insertion_options[:variants]
    
    # creating a new sentence requires an insertion location and which article to use
    # it then generates a sentence where the word at location i is given the determined article
    sentence_i = lambda insertion_index, article_index: ' '.join(
        word if insertion_index != current_index else f'{articles[article_index]} {word}'
        for current_index, word in enumerate(sentence)
        )
    
    # each of the constructed sentences has a predefined insertion location and article index
    return {sentence_i(insertion_index, article_index) 
            for insertion_index, article_index in selected_options}

    
def valid_random_article_insertion(sentence:Sequence[str], 
                             articles:tuple[str,...],
                             is_valid_sentence:Callable[[str],bool],
                             variants:int=3) -> Collection[str]:
    """
    This method computes a collection of random article inserted sentences.
    The maximum number of sentences is equal to the length of the sentence * articles supplied.
    As such the variants returned will be from this collection.    
    
    Parameters
    ----------
    sentence : Sequence[str]
        The sentence in a split set of words.
    articles : tuple[str,...]
        The articles to insert at random locations
    is_valid_sentence: Callable[[str], bool]
        Whether the sentence is a valid one.
    variants : int, optional
        How many alternatives ought to be computed, if None is passed all options are used. The default is 3.

    Returns
    -------
    Collection[str]
        Sentences of words with a random selected article at random

    """
    #print('-'*50)
    #print(sentence)
    #print('-'*50)
    # first create a collection locations with articles as tuples (this encompasses all possible solutions)
    insertion_options = list( 
        product([i for i in range(len(sentence))],
                [i for i in range(len(articles))])
        )
    # shuffle these locations
    random.shuffle(insertion_options)
    
    
    # creating a new sentence requires an insertion location and which article to use
    # it then generates a sentence where the word at location i is given the determined article
    sentence_i = lambda insertion_index, article_index: ' '.join(
        word if insertion_index != current_index else f'{articles[article_index]} {word}'
        for current_index, word in enumerate(sentence)
        )
    
    potential_sentences = (sentence_i(insertion_index, article_index) 
            for insertion_index, article_index in insertion_options)
    
    valid_sentences = filter(is_valid_sentence, potential_sentences)
    
    return set(islice(valid_sentences, variants))



def _standard_tests():
    print("Testing article insertion.")
    print('-'*20)
    test_zin = "maken van fietsen."
    articles = ("de", "het")
    absolute_maximum_options = len(test_zin.split()) * len(articles)
    
    number_of_elements = 3 # used for subsets
    
    print("Running all options test.")
    all_options = random_article_insertion(test_zin.split(), 
                                           articles,
                                           variants=None)
    expected_options = absolute_maximum_options
    print("Results:")
    print(test_zin, '-->', all_options)
    assert len(all_options) == expected_options, f"Incorrect number of options. Got {len(all_options)}, expected {expected_options}."
    print('-'*20)
    print("Running subset generation test.")    
    if number_of_elements >= absolute_maximum_options:
        print("Given subset will test against full returnable options, please reduce desired subset variable.")
        
    subset_options = random_article_insertion(test_zin.split(), articles, 
                                              variants=number_of_elements)
    expected_options = min(number_of_elements, absolute_maximum_options)
    
    print(f"Results for subset{number_of_elements}:")
    print(test_zin, '-->', subset_options)
    assert len(subset_options) == expected_options, f"Subset generation mismatch. Got {len(subset_options)}, expected {expected_options}."
    
    overstated_sample_count = absolute_maximum_options + 1
    overstated_subset = random_article_insertion(test_zin.split(), articles, 
                                              variants=overstated_sample_count)
    print(f"Results for subset {overstated_sample_count}:")
    print(test_zin, '-->', overstated_subset)
    assert len(overstated_subset) == absolute_maximum_options, "Ran into strange condition for larger than possible subset variable."
    

def _valid_sentence_tests():
    from valid_sentence_checker import valid_dutch, valid_english
    
    test_zin = "maken van fietsen."
    articles = ("de", "het")
    
    print("Running all options test.")
    all_options = valid_random_article_insertion(test_zin.split(), 
                                           articles, valid_dutch,
                                           variants=None)
    print(all_options)
    
    
    english_test = "researching production of microplastics"
    articles = ("the", 'a')
    all_options = valid_random_article_insertion(english_test.split(), 
                                           articles, valid_english,
                                           variants=None)
    print(all_options)
    
    english_test = "generally speaking, would you say that most people can be trusted, or that you can't be too careful in dealing with people?"
    articles = ("the", 'a')
    all_options = valid_random_article_insertion(english_test.split(), 
                                           articles, valid_english,
                                           variants=None)
    print(all_options)
    
    

if __name__ == "__main__":
    #_standard_tests()
    _valid_sentence_tests()