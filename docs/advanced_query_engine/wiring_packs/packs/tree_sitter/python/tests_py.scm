; Pytest/unittest test discovery (name-based)
; Intended execution: QueryCursor.matches(...)

; pytest function tests: def test_*(...)
(function_definition
  name: (identifier) @test.func.name
) @test.func.node
(#match? @test.func.name "^test_")

; unittest style: class Test*(...)
(class_definition
  name: (identifier) @test.class.name
) @test.class.node
(#match? @test.class.name "^Test")

; unittest methods: def test_*(self, ...)
(function_definition
  name: (identifier) @test.method.name
) @test.method.node
(#match? @test.method.name "^test_")
