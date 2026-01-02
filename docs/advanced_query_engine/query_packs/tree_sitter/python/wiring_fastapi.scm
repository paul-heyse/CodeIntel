; FastAPI wiring (route decorators + add_api_route)
; Intended execution: QueryCursor.matches(...)
;
; Captures:
;   - @wire.decorator_call: the decorator's call node
;   - @wire.method: HTTP method attr (get/post/...)
;   - @wire.path: first arg string node
;   - @wire.handler_name: function name
;
; NOTE: captures are intentionally "one per capture name per match".

; @router.get("/path") def handler(...)
(decorated_definition
  (decorator
    (call
      function: (attribute
        object: (identifier) @wire.registrar
        attribute: (identifier) @wire.method
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
(#match? @wire.method "^(get|post|put|patch|delete|options|head|websocket)$")

; app.add_api_route("/path", handler, ...)
(call
  function: (attribute
    object: (identifier) @wire.registrar2
    attribute: (identifier) @wire.add_api_route
  )
  arguments: (argument_list
    (string) @wire.path2
    (_) @wire.handler_expr2
    .
  ) @wire.add_api_route_args
) @wire.add_api_route_call
(#eq? @wire.add_api_route "add_api_route")
