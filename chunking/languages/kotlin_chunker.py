"""Kotlin-specific tree-sitter based chunker."""

from typing import Any, Dict, List, Optional, Set

from chunking.base_chunker import LanguageChunker


class KotlinChunker(LanguageChunker):
    """Kotlin-specific chunker using tree-sitter."""

    def __init__(self):
        super().__init__('kotlin')

    def _get_splittable_node_types(self) -> Set[str]:
        """Kotlin-specific splittable node types."""
        return {
            'class_declaration',
            'function_declaration',
            'secondary_constructor',
            'companion_object',
            'object_declaration',
            'property_declaration',
            'anonymous_initializer',  # init { } blocks
        }

    def _extract_kdoc(self, node: Any, source: bytes) -> Optional[str]:
        """Extract a preceding KDoc or block comment as a docstring.

        Kotlin doc comments (/** ... */) appear as the previous named sibling
        of the declaration node in the parse tree.
        """
        prev = node.prev_named_sibling
        if prev and prev.type in ('block_comment', 'line_comment'):
            return self.get_node_text(prev, source).strip()
        return None

    def _parse_modifiers(self, node: Any, source: bytes) -> Dict[str, Any]:
        """Parse the modifiers child of a node into annotations, class modifiers, and function modifiers."""
        result: Dict[str, Any] = {}
        for child in node.children:
            if child.type != 'modifiers':
                continue
            annotations: List[str] = []
            class_mods: List[str] = []
            fn_mods: List[str] = []
            plain_mods: List[str] = []
            for modifier in child.children:
                if modifier.type == 'annotation':
                    annotations.append(self.get_node_text(modifier, source))
                elif modifier.type == 'class_modifier':
                    text = self.get_node_text(modifier, source).strip()
                    class_mods.append(text)
                    plain_mods.append(text)
                elif modifier.type in ('function_modifier', 'member_modifier',
                                       'visibility_modifier', 'inheritance_modifier',
                                       'property_modifier'):
                    text = self.get_node_text(modifier, source).strip()
                    fn_mods.append(text)
                    plain_mods.append(text)
                elif modifier.is_named:
                    plain_mods.append(self.get_node_text(modifier, source).strip())
            if annotations:
                result['annotations'] = annotations
            if class_mods:
                result['class_modifiers'] = class_mods
            if fn_mods:
                result['function_modifiers'] = fn_mods
            if plain_mods:
                result['modifiers'] = plain_mods
            break  # Only the first modifiers node matters
        return result

    def extract_metadata(self, node: Any, source: bytes) -> Dict[str, Any]:
        """Extract Kotlin-specific metadata."""
        metadata: Dict[str, Any] = {'node_type': node.type}

        # KDoc / block comment immediately preceding this declaration
        kdoc = self._extract_kdoc(node, source)
        if kdoc:
            metadata['docstring'] = kdoc

        # Name: first direct identifier child
        for child in node.children:
            if child.type == 'identifier':
                metadata['name'] = self.get_node_text(child, source)
                break

        # Modifiers: annotations, class_modifiers, function_modifiers
        metadata.update(self._parse_modifiers(node, source))

        # Node-type–specific enrichment
        if node.type == 'class_declaration':
            keyword_tokens = {
                self.get_node_text(child, source)
                for child in node.children
                if not child.is_named
            }
            class_mods = metadata.get('class_modifiers', [])
            if 'interface' in keyword_tokens:
                metadata['declaration_kind'] = 'interface'
            elif 'enum' in class_mods:
                metadata['declaration_kind'] = 'enum'
            else:
                metadata['declaration_kind'] = 'class'
            if any(child.type == 'type_parameters' for child in node.children):
                metadata['has_generics'] = True

        elif node.type == 'function_declaration':
            children = node.children
            child_texts_anon = [
                self.get_node_text(c, source)
                for c in children
                if not c.is_named
            ]
            # Extension function: a '.' appears as an anonymous sibling before the identifier
            if '.' in child_texts_anon:
                for child in children:
                    if child.type in ('user_type', 'nullable_type'):
                        metadata['receiver_type'] = self.get_node_text(child, source)
                        metadata['is_extension'] = True
                        break

            # Return type: user_type or nullable_type after ':' that follows function_value_parameters
            saw_params = False
            saw_colon = False
            for child in children:
                if child.type == 'function_value_parameters':
                    saw_params = True
                elif saw_params and not child.is_named and self.get_node_text(child, source) == ':':
                    saw_colon = True
                elif saw_colon and child.type in ('user_type', 'nullable_type'):
                    metadata['return_type'] = self.get_node_text(child, source)
                    break

            # Generics (type_parameters before function_value_parameters)
            if any(child.type == 'type_parameters' for child in children):
                metadata['has_generics'] = True

            # Suspend functions are Kotlin's async primitive; surface as is_async
            # so the existing 'async' tag path in multi_language_chunker picks it up.
            if 'suspend' in metadata.get('function_modifiers', []):
                metadata['is_async'] = True

        elif node.type in ('object_declaration', 'companion_object'):
            metadata['declaration_kind'] = 'object'

        elif node.type == 'property_declaration':
            metadata['declaration_kind'] = 'property'
            for child in node.children:
                if child.type == 'variable_declaration':
                    for subchild in child.children:
                        if subchild.type == 'identifier':
                            metadata['name'] = self.get_node_text(subchild, source)
                            break
                    break

        elif node.type == 'anonymous_initializer':
            metadata['name'] = '<init block>'
            metadata['declaration_kind'] = 'init'

        return metadata
