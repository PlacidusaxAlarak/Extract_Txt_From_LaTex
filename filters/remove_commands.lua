-- filters/remove_commands.lua
--
-- A Pandoc Lua filter to remove specific raw LaTeX commands or environments
-- that are known to cause issues during conversion.

-- Add the names of commands (including backslash) or environments to this list.
local unsupported_raw_tex = {
  "\\mycustomcommand", -- Example: a command to remove
  "specialfigure"      -- Example: an environment to remove
}

-- Create a set for faster lookups.
local to_remove = {}
for _, item in ipairs(unsupported_raw_tex) do
  to_remove[item] = true
end

-- Function to process raw inline TeX.
-- This handles single commands like \mycustomcommand{...}
function RawInline(el)
  -- el.text contains the raw TeX, e.g., "\mycustomcommand{some value}"
  -- A simple check: does it start with a known unsupported command?
  for tex, _ in pairs(to_remove) do
    if el.text:find(tex, 1, true) == 1 then
      -- Returning an empty list of inlines effectively deletes the element.
      return {}
    end
  end
  -- Otherwise, keep the element as is.
  return el
end

-- Function to process raw TeX blocks.
-- This handles environments like \begin{specialfigure}...\end{specialfigure}
function RawBlock(el)
  -- el.text contains the raw TeX, e.g., "\begin{specialfigure}..."
  -- A simple check: does it start with \begin{...} of a known unsupported environment?
  for tex, _ in pairs(to_remove) do
    local pattern = "\begin{" .. tex .. "}"
    if el.text:find(pattern, 1, true) == 1 then
      -- Returning an empty list of blocks effectively deletes the element.
      return {}
    end
  end
  -- Otherwise, keep the element as is.
  return el
end
