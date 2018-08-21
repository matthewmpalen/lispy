class Symbol(str):
    pass


def get_symbol(s, symbol_table={}):
    if s not in symbol_table:
        symbol_table[s] = Symbol(s)

    return symbol_table[s]


# Built-in symbols

quote = get_symbol('quote')
if_ = get_symbol('if')
set_ = get_symbol('set!')
define = get_symbol('define')
lambda_ = get_symbol('lambda')
begin = get_symbol('begin')
define_macro = get_symbol('define-macro')
quasiquote = get_symbol('quasiquote')
unquote = get_symbol('unquote')
unquote_splicing = get_symbol('unquote-splicing')

append = get_symbol('append')
cons_ = get_symbol('cons')
let = get_symbol('let')


# Note: uninterned; can't be read
eof_object = Symbol('#<eof-object>')

quotes = {
    "'": quote,
    "`": quasiquote,
    ",": unquote,
    ",@": unquote_splicing
}