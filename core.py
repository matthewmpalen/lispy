from io import IOBase, StringIO
import re
import sys

from symbols import *


class InPort(object):
    """
    An input port. Retains a line of chars.
    """

    TOKENIZER = r"""\s*(,@|[('`,)]|"(?:[\\].|[^\\"])*"|;.*|[^\s('"`,;)]*)(.*)"""

    def __init__(self, file):
        self.file = file
        self.line = ''

    def next_token(self):
        """
        Return the next token, reading new text into line buffer if needed.

        :return: str
        """

        while True:
            if self.line == '':
                self.line = self.file.readline()

            if self.line == '':
                return eof_object

            token, self.line = re.match(InPort.TOKENIZER, self.line).groups()
            if token != '' and not token.startswith(';'):
                return token


def parse(inport):
    """
    Parse a program: read and expand/error-check it.
    (Backwards compatibility: given a str, convert it to an InPort)

    :param inport:
    :return:
    """

    if isinstance(inport, str):
        inport = InPort(StringIO(inport))

    return expand(read(inport), top_level=True)


def atomize(token):
    """
    Numbers become numbers; #t and #f are booleans; "..." string; otherwise Symbol.

    :param token: str
    :return: Object
    """

    if token == '#t':
        return True
    elif token == '#f':
        return False
    elif token[0] == '"':
        return token[1:-1].decode('string_escape')

    try:
        return int(token)
    except ValueError:
        try:
            return float(token)
        except ValueError:
            try:
                return complex(token.replace('i', 'j', 1))
            except ValueError:
                return get_symbol(token)


def readchar(inport):
    """
    Read the next character from an input port.

    :param inport:
    :return: char
    """

    if inport.line != '':
        # Removes first character from inport.line
        ch, inport.line = inport.line[0], inport.line[1:]
        return ch
    else:
        return inport.file.read(1) or eof_object


def read(inport):
    """
    Read a Scheme expression from an input port.

    :param inport:
    :return: atom
    """

    def read_ahead(token):
        if '(' == token:
            lst = []
            while True:
                token = inport.next_token()
                if token == ')':
                    return lst
                else:
                    lst.append(read_ahead(token))
        elif ')' == token:
            raise SyntaxError('unexpected )')
        elif token in quotes:
            return [quotes[token], read(inport)]
        elif token is eof_object:
            raise SyntaxError('unexpected EOF in list')
        else:
            return atomize(token)

    # body of read:
    token1 = inport.next_token()
    if token1 is eof_object:
        return eof_object

    return read_ahead(token1)


class Env(dict):
    """
    An environment: a dict of {'var':val} pairs, with an outer Env.
    """

    def __init__(self, params=(), args=(), outer=None):
        # Bind param list to corresponding args, or single parm to list of args

        self.outer = outer

        if isinstance(params, Symbol):
            self.update({
                params: list(args)
            })
        else:
            if len(args) != len(params):
                raise TypeError('expected {}, given {}'.format(
                    to_string(params),
                    to_string(args))
                )

            self.update(zip(params, args))

    def find(self, var):
        """
        Find the innermost Env where var appears.

        :param var: Object
        :return: Env
        """

        if var in self:
            return self
        elif self.outer is None:
            raise LookupError(var)
        else:
            return self.outer.find(var)


class Procedure:
    """
    A user-defined Scheme procedure.
    """

    def __init__(self, params, exp, env):
        self.params = params
        self.exp = exp
        self.env = env

    def __call__(self, *args):
        return eval_(
            self.exp,
            Env(self.params, args, self.env)
        )


def to_string(x):
    """
    Convert a Python object back into a Lisp-readable string.

    :param x: Object
    :return:  str
    """

    if x is True:
        return '#t'
    elif x is False:
        return '#f'
    elif isinstance(x, Symbol):
        return x
    elif isinstance(x, str):
        return '"{}"'.format(x.encode('string_escape').replace('"',r'\"'))
    elif isinstance(x, list):
        # Technically should use paren rather than square brackets
        return str(x)
    elif isinstance(x, complex):
        return str(x).replace('j', 'i')
    else:
        return str(x)


def repl(prompt='lispy> ', inport=InPort(sys.stdin), out=sys.stdout):
    """
    "A prompt-read-eval-print loop."

    :param prompt: str
    :param inport: STDIN or File
    :param out:
    :return:
    """

    sys.stderr.write("Lispy version 2.x\n")

    while True:
        try:
            if prompt:
                sys.stderr.write(prompt)

            x = parse(inport)
            if x is eof_object:
                return

            val = eval_(x)
            if val is not None and out:
                print(to_string(val), file=out)
        except Exception as e:
            print('{}: {}'.format(type(e).__name__, e))


def load(filename):
    """
    Eval every expression from a file.

    :param filename: str
    :return:
    """

    with open(filename) as f:
        repl(None, InPort(f), None)


def is_pair(x):
    return x != [] and isinstance(x, list)


def cons(x, y):
    return [x] + y


def callcc(proc):
    """
    Call procedure with current continuation; escape only

    :param proc: Procedure
    :return:
    """

    ball = RuntimeWarning("Sorry, can't continue this continuation any longer.")

    def throw(retval):
        ball.retval = retval
        raise ball

    try:
        return proc(throw)
    except RuntimeWarning as w:
        if w is ball:
            return ball.retval
        else:
            raise w


def add_globals(self):
    """
    Add some Scheme standard procedures to environment
    :param self:
    :return:
    """

    import math, cmath, operator as op

    self.update(vars(math))
    self.update(vars(cmath))

    extras = {
        '+': op.add,
        '-': op.sub,
        '*': op.mul,
        '/': op.truediv,
        'not': op.not_,
        '>': op.gt,
        '<': op.lt,
        '>=': op.ge,
        '<=': op.le,
        '=': op.eq,
        'equal?': op.eq,
        'eq?': op.is_,
        'length': len,
        'cons': cons,
        'car': lambda x: x[0],
        'cdr': lambda x: x[1:],
        'append': op.add,
        'list': lambda *x: list(x),
        'list?': lambda x: isinstance(x,list),
        'null?': lambda x: x == [],
        'symbol?': lambda x: isinstance(x, Symbol),
        'boolean?': lambda x: isinstance(x, bool),
        'pair?': is_pair,
        'port?': lambda x: isinstance(x, IOBase),
        'apply':lambda proc, l: proc(*l),
        'eval': lambda x: eval_(expand(x)),
        'load': lambda fn: load(fn),
        'call/cc': callcc,
        'open-input-file': open,
        'close-input-port': lambda p: p.file.close(),
        'open-output-file': lambda f:open(f,'w'),
        'close-output-port': lambda p: p.close(),
        'eof-object?': lambda x: x is eof_object,
        'read-char': readchar,
        'read': read,
        'write': lambda x, port=sys.stdout: port.write(to_string(x)),
        'display': lambda x, port=sys.stdout: port.write(x if isinstance(x, str) else to_string(x))
    }

    self.update(extras)
    return self


global_env = add_globals(Env())


def eval_(x, env=global_env):
    """
    Evaluate an expression in an environment.

    :param x: Object
    :param env: Env
    :return:
    """

    while True:
        if isinstance(x, Symbol):
            # variable reference
            return env.find(x)[x]
        elif not isinstance(x, list):
            # constant literal
            return x
        elif x[0] is quote:
            # (quote exp)
            (_, exp) = x
            return exp
        elif x[0] is if_:
            # (if test conseq alt)
            (_, test, conseq, alt) = x
            x = (conseq if eval_(test, env) else alt)
        elif x[0] is set_:
            # (set! var exp)
            (_, var, exp) = x
            env.find(var)[var] = eval_(exp, env)
            return None
        elif x[0] is define:
            # (define var exp)
            (_, var, exp) = x
            env[var] = eval_(exp, env)
            return None
        elif x[0] is lambda_:
            # (lambda (var*) exp)
            (_, vars, exp) = x
            return Procedure(vars, exp, env)
        elif x[0] is begin:
            # (begin exp+)
            for exp in x[1:-1]:
                eval_(exp, env)

            x = x[-1]
        else:
            # (proc exp*)
            exps = [eval_(exp, env) for exp in x]

            proc = exps.pop(0)
            if isinstance(proc, Procedure):
                x = proc.exp
                env = Env(proc.params, exps, proc.env)
            else:
                return proc(*exps)


def expand(x, top_level=False):
    """
    Walk tree of x, making optimizations/fixes, and signaling SyntaxError.

    :param x:
    :param top_level: bool
    :return:
    """

    # () => Error
    require(x, x != [])
    if not isinstance(x, list):
        # constant => unchanged
        return x
    elif x[0] is quote:
        # (quote exp)
        require(x, len(x) == 2)
        return x
    elif x[0] is if_:
        # (if t c) => (if t c None)
        if len(x) == 3:
            x += [None]

        require(x, len(x) == 4)
        return list(map(expand, x))
    elif x[0] is set_:
        require(x, len(x) == 3)
        var = x[1]

        # (set! non-var exp) => Error
        require(x, isinstance(var, Symbol), "can set! only a symbol")
        return [set_, var, expand(x[2])]
    elif x[0] is define or x[0] is define_macro:
        require(x, len(x) >= 3)

        # (define (f args) body)
        def_, v, body = x[0], x[1], x[2:]
        if isinstance(v, list) and v:
            #  => (define f (lambda (args) body))
            f, args = v[0], v[1:]
            return expand([def_, f, [lambda_, args] + body])
        else:
            # (define non-var/list exp) => Error
            require(x, len(x) == 3)
            require(x, isinstance(v, Symbol), "can define only a symbol")
            exp = expand(x[2])

            if def_ is define_macro:
                require(x, top_level, "define-macro only allowed at top level")
                proc = eval_(exp)
                require(x, callable(proc), "macro must be a procedure")
                # (define-macro v proc)
                macro_table[v] = proc
                #  => None; add v:proc to macro_table
                return None

            return [define, v, exp]
    elif x[0] is begin:
        if len(x) == 1:
            # (begin) => None
            return None
        else:
            return [expand(xi, top_level) for xi in x]
    elif x[0] is lambda_:
        # (lambda (x) e1 e2)
        require(x, len(x) >= 3)
        #  => (lambda (x) (begin e1 e2))
        vars_, body = x[1], x[2:]

        predicate = (
            (isinstance(vars_, list) and all(isinstance(v, Symbol) for v in vars_))
            or isinstance(vars_, Symbol)
        )
        require(x, predicate, "illegal lambda argument list")
        exp = body[0] if len(body) == 1 else [begin] + body
        return [lambda_, vars_, expand(exp)]
    elif x[0] is quasiquote:
        # `x => expand_quasiquote(x)
        require(x, len(x) == 2)
        return expand_quasiquote(x[1])
    elif isinstance(x[0], Symbol) and x[0] in macro_table:
        # (m arg...)
        return expand(macro_table[x[0]](*x[1:]), top_level)
    else:
        #        => macroexpand if m isa macro
        # (f arg...) => expand each
        return list(map(expand, x))


def require(x, predicate, msg="wrong length"):
    """
    Signal a syntax error if predicate is false.

    :param x: Object
    :param predicate: Object
    :param msg: str
    :return:
    """

    if not predicate:
        raise SyntaxError(to_string(x) + ': ' + msg)


def expand_quasiquote(x):
    """
    Expand `x => 'x; `,x => x; `(,@x y) => (append x y)

    :param x:
    :return: list
    """

    if not is_pair(x):
        return [quote, x]

    require(x, x[0] is not unquote_splicing, "can't splice here")
    if x[0] is unquote:
        require(x, len(x) == 2)
        return x[1]
    elif is_pair(x[0]) and x[0][0] is unquote_splicing:
        require(x[0], len(x[0]) == 2)
        return [
            append,
            x[0][1],
            expand_quasiquote(x[1:])
        ]
    else:
        return [
            cons_,
            expand_quasiquote(x[0]),
            expand_quasiquote(x[1:])
        ]


def let(*args):
    args = list(args)
    x = cons(let, args)
    require(x, len(args) > 1)

    bindings, body = args[0], args[1:]
    require(
        x,
        all(
            isinstance(b, list)
            and len(b) == 2
            and isinstance(b[0], Symbol) for b in bindings
        ),
        "illegal binding list"
    )

    vars_, vals = zip(*bindings)
    return [
               [
                   lambda_,
                   list(vars_)
               ] + list(map(expand, body))
           ] + list(map(expand, vals))


# More macros can go here
macro_table = {let: let}
