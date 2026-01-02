; Python calls (captures callee expression + argument list)
; Intended execution: QueryCursor.matches(...)

(call
  function: (_) @call.callee
  arguments: (argument_list) @call.args
) @call.node
