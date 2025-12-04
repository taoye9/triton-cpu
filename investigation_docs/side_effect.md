# Side Effects in MLIR

Operations can have implicit behavior not represented by SSA data flow (e.g., memory reads/writes, device I/O, UB). These must be modeled explicitly so compiler passes don’t reorder, eliminate, or introduce them incorrectly.

**Core interfaces**
- **MemoryEffectsOpInterface**: declares how an op reads/writes memory or abstract resources.
- **ConditionallySpeculatable**: declares when an op’s behavior is undefined or may not terminate, controlling speculation/hoisting.

**Effect scopes**
- **Operand-scoped**: effect tied to a specific operand (enables precise alias analysis). Use builder `EffectInstance(Effect*, OpOperand*, Resource*)`.
- **Operation-scoped**: effect on a broader resource (global/abstract, not tied to a single operand). Use builder `EffectInstance(Effect*, Resource*)`. Example: a volatile store that must not be reordered.

**External Interface Usage**

- Used by core passes: DCE (dead-code), CSE (common-subexpr), LICM (loop-invariant code motion).
- Pattern: query `MemoryEffectOpInterface` to decide safety of elimination/hoisting.

```cpp
if (auto iface = dyn_cast<MemoryEffectOpInterface>(op)) {
  const bool hasSideEffects =
      iface.hasEffect<MemoryEffects::Write>() ||
      iface.hasEffect<MemoryEffects::Read>()  ||
      iface.hasEffect<MemoryEffects::Allocate>() ||
      iface.hasEffect<MemoryEffects::Free>();

  if (!hasSideEffects)
    markForDeletion(op);
}
```


## Declaring Memory Effects in ODS (TableGen)

### **Op-level (manual `getEffects()`)**
- Add `DeclareOpInterfaceMethods<MemoryEffectsOpInterface>` to your op’s trait list.
- Implement `getEffects(...)` in C++ when effects are complex or involve custom resources.

Example:
```tablegen
def LoadOp : Triton_Op<"load", [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "Load from a memref";
  let description = [{ Loads a value from a memref at a given index. }];
}
```

Traits and interfaces are distinct in ODS:
- **Traits** belong in the trait list: `Op<..., [Pure, Elementwise]>`.
- **Interfaces** must be declared via `DeclareOpInterfaceMethods<...>`; they are not traits and cannot appear directly in the trait list.



### **Operand-level/Abstract (auto-generated `getEffects()`)**
- Annotate operands with side-effect traits; ODS synthesizes `getEffects()`.

Example:
```tablegen
ins Arg<AnyMemRef, "source memref", [MemReadAt<0, FullEffect>]>:$source
```
- `Arg<Constraint, "doc", [annotations]>:$name` declares an operand.
- `Constraint` type constrain specifies the operand type (e.g., `AnyMemRef`, `TT_PtrLike`).
- `annotations` attach operand-scoped metadata (e.g., `MemReadAt`, `MemWriteAt`).
- `$name` is the operand’s SSA name used in the op definition.

Contrast with other wrappers:
- `Optional<Constraint>`: an operand that may be absent. Effects tied to an optional operand should guard on presence in `getEffects()` if you implement it manually.
- `Variadic<Constraint>`: a list of operands of the same kind. When annotating variadics, document the intended coverage (e.g., read all, write all, or per-element semantics) and prefer operand-scoped annotations at the group level.

Supported operand annotations:
- `MemReadAt<stage, effect>`
- `MemWriteAt<stage, effect>`
    - Lower stage runs earlier: `MemReadAt<0, ...>` before `MemWriteAt<1, ...>`.
    - Use increasing stages to express sequencing (read→compute→write, write→fence, etc.).
- `MemRead`: `MemReadAt<0, FullEffect>`
- `MemWrite`: `MemWriteAt<0, FullEffect>`

some subtleties about effect:
- the kind of memory effect
(MemoryEffects::Read, MemoryEffects::Write, MemoryEffects::Allocate, MemoryEffects::Free)

- the extent of the memory that the effect touches, i.e. the “effect region” (FullEffect, PartialEffect)


## Design Checklist for New Ops

Ask these questions:
- **Reads/writes memory?** Implement `MemoryEffectsOpInterface`.
- **Effect ordering matters?** Model stages/ordering to improve analysis precision.
- **Affects entire resource?** Use `FullEffect` where appropriate.
- **Must-preserve effects?** Volatile stores, syscalls, device I/O: use `MemoryEffectsOpInterface` with abstract `Resource`. If the effect is novel, consider an RFC.
- **Runtime preconditions?** Implement `ConditionallySpeculatable`.
- **Possible non-termination?** Implement `ConditionallySpeculatable`.
- **Non-local control flow (e.g., `longjmp`)?** Not well modeled yet; patches welcome.
- **No side effects?** Mark the op `Pure` so it may be hoisted/introduced/eliminated (MLIR’s `Pure` indicates absence of side effects).
