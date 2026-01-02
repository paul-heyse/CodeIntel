; Flask wiring (route decorators + add_url_rule)
; Intended execution: QueryCursor.matches(...)

; @app.route("/path") def handler(...)
(decorated_definition
  (decorator
    (call
      function: (attribute
        object: (identifier) @wire.registrar
        attribute: (identifier) @wire.route_attr
      )
      arguments: (argument_list
        (string) @wire.path
        .
      ) @wire.decorator_args
    ) @wire.decorator_call
  )
  (function_definition
    name: (identifier) @wire.handler_name
  ) @wire.handler_def
) @wire.decorated_def
(#eq? @wire.route_attr "route")

; app.add_url_rule("/path", endpoint, view_func=handler, ...)
(call
  function: (attribute
    object: (identifier) @wire.registrar2
    attribute: (identifier) @wire.add_url_rule
  )
  arguments: (argument_list
    (string) @wire.path2
    .
  ) @wire.add_url_rule_args
) @wire.add_url_rule_call
(#eq? @wire.add_url_rule "add_url_rule")
