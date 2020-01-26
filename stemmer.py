# This stemmer is implemented on the ideas of the stemmer for Croatian language originally
# designed in the Science Department of the Faculty of Philosophy, University of Zagreb

import re
import sys

STOP_WORDS =set([word.strip() for word in open('stop_words.txt')])

PREDEFINED_RULES = [re.compile(r'^('+osnova+')('+nastavak+r')$') for osnova, nastavak in [e.strip().split(' ') for e in open('rules.txt', encoding='utf8')]]
PREDEFINED_TRANSFORMATIONS = [e.strip().split('\t') for e in open('transformations.txt', encoding='utf8')]

CROATIAN_TO_ENGLISH_SYMBOLS = {
    'š' : 's',
    'ć' : 'c',
    'č' : 'c',
    'ž' : 'z',
    'đ' : 'd'
}

def amplify_R(niz):
    '''
    The letter R is considered to be a special kind of vowel in Croatian.
    '''
    return re.sub(r'(^|[^aeiou])r($|[^aeiou])',r'\1R\2',niz)

def has_vowel(word):
    '''
    Checks whether there is a vowel in the word.
    '''
    if re.search(r'[aeiouR]',amplify_R(word)) is None:
    	return False
    else:
    	return True

def transform(word):
    '''
    The method that does tranformation (similar to lemmatization) of the word to its default case system. 
    '''
    for orig,sub in PREDEFINED_TRANSFORMATIONS:
        if word.endswith(orig):
            return word[:-len(orig)]+sub
    return word

def convert_to_root(word):
    '''
    The stemming method.
    '''
    for rule in PREDEFINED_RULES:
        rule_split = rule.match(word)
        if rule_split is not None:
            if has_vowel(rule_split.group(1)) and len(rule_split.group(1))>1:
                return rule_split.group(1)
    return word

def remove_croatian_symbols(token):
    '''
    Removes Croatian symbols from the words.
    '''
    for key, value in CROATIAN_TO_ENGLISH_SYMBOLS.items():
        token = token.replace(key, value)

    return token