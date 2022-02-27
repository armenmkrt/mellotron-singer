""" from https://github.com/keithito/tacotron """
import re

from g2p_en.g2p import G2p
from phonemizer.phonemize import phonemize

from data.text import cleaners
from data.text.symbols import g2p_phonemes
from data.text.symbols import ipa_phonemes

# G2p phonemizer for inference
g2p = G2p()

# Regular expression matching text enclosed in curly braces:
_curly_re = re.compile(r'(.*?)\{(.+?)\}(.*)')


# Mappings from symbol to numeric ID and vice versa:
def get_symbol_id_dicts(ipa_chars, g2p_chars, is_g2p=True):
    if is_g2p:
        symbol_to_id = {s: i for i, s in enumerate(g2p_chars)}
        id_to_symbol = {i: s for i, s in enumerate(g2p_chars)}
    else:
        symbol_to_id = {s: i for i, s in enumerate(ipa_chars)}
        id_to_symbol = {i: s for i, s in enumerate(ipa_chars)}
    return symbol_to_id, id_to_symbol


def phoneme_duration_to_sequence(duration_phoneme, is_g2p=True):
    duration_data = []
    phoneme_data = []
    _symbol_to_id, _ = get_symbol_id_dicts(ipa_phonemes, g2p_phonemes, is_g2p)
    for item in duration_phoneme:
        duration_data.append(float(item.split(" ")[0]))
        phoneme_data.append(_symbol_to_id[item.split(" ")[1]])

    return phoneme_data, duration_data


def text_to_phonemized_sequence(text: str, cleaner_names, is_g2p=True):
    """
    Converts a string of text to a sequence of IDs corresponding to the symbols in the text.

    :param text: text representation of input
    :param cleaner_names:
    :param is_g2p:

    :return: list of converted token IDs
    """
    _symbol_to_id, _ = get_symbol_id_dicts(ipa_phonemes, g2p_phonemes, is_g2p)
    print(_symbol_to_id)
    if is_g2p:
        sequence, phonemized_symbols = _g2p_phonemized_symbol_to_sequence(text, _symbol_to_id)
        # Append EOS token
        sequence.append(_symbol_to_id["<"])
    else:
        sequence, phonemized_symbols = _ipa_phonemized_symbol_to_sequence(text, _symbol_to_id)
        # Append EOS, SOS tokens
        sequence.insert(0, _symbol_to_id['>'])
        sequence.append(_symbol_to_id['<'])

        phonemized_symbols = ">" + phonemized_symbols + "<"
    print(sequence)
    return sequence, phonemized_symbols


def text_to_sequence(text, cleaner_names):
    """
    Converts a string of text to a sequence of IDs corresponding to the symbols in the text.

    The text can optionally have ARPAbet sequences enclosed in curly braces embedded
    in it. For example, "Turn left on {HH AW1 S S T AH0 N} Street."

    Args:
        text: string to convert to a sequence
        cleaner_names: names of the cleaner functions to run the text through

    Returns:
        List of integers corresponding to the symbols in the text
    """
    sequence = []
    _symbol_to_id, _ = get_symbol_id_dicts(g2p_phonemes, g2p_chars=ipa_phonemes, is_g2p=True)
    # Check for curly braces and treat their contents as ARPAbet:
    while len(text):
        m = _curly_re.match(text)
        if not m:
            sequence += _symbols_to_sequence(_clean_text(text, cleaner_names))
            break
        sequence += _symbols_to_sequence(_clean_text(m.group(1), cleaner_names))
        sequence += _arpabet_to_sequence(m.group(2), _symbol_to_id)
        text = m.group(3)

    return sequence


def _clean_text(text, cleaner_names):
    for name in cleaner_names:
        cleaner = getattr(cleaners, name)
        if not cleaner:
            raise Exception('Unknown cleaner: %s' % name)
        text = cleaner(text)
    return text


def _symbols_to_sequence(symbols, symbol_to_id):
    return [symbol_to_id[s] for s in symbols if _should_keep_symbol(s, symbol_to_id)]


def _g2p_phonemized_symbol_to_sequence(symbols, symbol_to_id):
    phonemized_symbols = g2p(symbols)
    phonemized_symbols = [symbol if symbol not in {".", ","} else "sp" for symbol in phonemized_symbols]
    results = [s for s in phonemized_symbols if _should_keep_symbol(s, symbol_to_id)]
    final_result = [symbol_to_id[result] for result in results]
    return final_result, phonemized_symbols


def _ipa_phonemized_symbol_to_sequence(sequence, symbol_to_id):
    phonemized_symbols = phonemize(sequence,
                                   preserve_punctuation=True,
                                   backend='espeak',
                                   strip=True,
                                   with_stress=False,
                                   language_switch='remove-flags')
                                   # punctuation_marks='!,-.:;? ()')
    final_result = [symbol_to_id[result] for result in phonemized_symbols]
    return final_result, phonemized_symbols


def _arpabet_to_sequence(text, symbol_to_id):
    return _symbols_to_sequence(['@' + s for s in text.split()], symbol_to_id)


def _should_keep_symbol(s, symbol_to_id):
    return s in symbol_to_id and (s != '_') and (s != '~')
