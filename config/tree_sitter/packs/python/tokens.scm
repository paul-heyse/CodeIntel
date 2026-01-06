; Identifier tokens
(identifier) @token.identifier

; Literal tokens
(string) @token.string
(integer) @token.number
(float) @token.number
(true) @token.boolean
(false) @token.boolean
(none) @token.none

; Keyword tokens (anonymous literals)
"def" @token.keyword
"class" @token.keyword
"if" @token.keyword
"elif" @token.keyword
"else" @token.keyword
"for" @token.keyword
"while" @token.keyword
"try" @token.keyword
"except" @token.keyword
"finally" @token.keyword
"with" @token.keyword
"async" @token.keyword
"await" @token.keyword
"return" @token.keyword
"yield" @token.keyword
"import" @token.keyword
"from" @token.keyword
"as" @token.keyword
"pass" @token.keyword
"break" @token.keyword
"continue" @token.keyword
"raise" @token.keyword
"global" @token.keyword
"nonlocal" @token.keyword
"lambda" @token.keyword
"assert" @token.keyword
"del" @token.keyword
"and" @token.keyword
"or" @token.keyword
"not" @token.keyword
"in" @token.keyword
"is" @token.keyword
