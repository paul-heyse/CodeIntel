; Python imports
; Intended execution: QueryCursor.matches(...)

(import_statement
  name: (dotted_name) @import.module
) @import.node

(import_from_statement
  module_name: (dotted_name) @import.from.module
  name: (dotted_name) @import.from.name
) @import.from.node

(import_from_statement
  module_name: (dotted_name) @import.from.module
  name: (wildcard_import) @import.from.wildcard
) @import.from.wild.node
