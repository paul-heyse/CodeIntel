; Python definitions (functions + classes + simple assignments)
; Intended execution: QueryCursor.matches(...) (unique capture names per pattern)

; --- Functions ---
(function_definition
  name: (identifier) @def.func.name
  parameters: (parameters) @def.func.params
  body: (block) @def.func.body
) @def.func.node

; --- Classes ---
(class_definition
  name: (identifier) @def.class.name
  superclasses: (argument_list)? @def.class.bases
  body: (block) @def.class.body
) @def.class.node

; --- Simple module-level assignment candidates ---
; NOTE: this also matches inside blocks; downstream should filter to module scope if needed.
(assignment
  left: (identifier) @def.assign.name
  right: (_) @def.assign.value
) @def.assign.node
