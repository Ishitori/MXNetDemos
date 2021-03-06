using	NO_KP
constructed	NO_KP
types	NO_KP
in	NO_KP
c++	BEGIN_KP
unions	NO_KP
the	NO_KP
c++	BEGIN_KP
standard	INSIDE_KP
states	NO_KP
that	NO_KP
a	NO_KP
union	BEGIN_KP
type	INSIDE_KP
cannot	NO_KP
have	NO_KP
a	NO_KP
member	NO_KP
with	NO_KP
a	NO_KP
nontrivial	NO_KP
constructor	NO_KP
or	NO_KP
destructor.	NO_KP
while	NO_KP
at	NO_KP
first	NO_KP
this	NO_KP
seems	NO_KP
unreasonable,	NO_KP
further	NO_KP
thought	NO_KP
makes	NO_KP
it	NO_KP
clear	NO_KP
why	NO_KP
this	NO_KP
is	NO_KP
the	NO_KP
case:	NO_KP
the	NO_KP
crux	NO_KP
of	NO_KP
the	NO_KP
problem	NO_KP
is	NO_KP
that	NO_KP
unions	NO_KP
don't	NO_KP
have	NO_KP
built-in	NO_KP
semantics	NO_KP
for	NO_KP
denoting	NO_KP
when	NO_KP
a	NO_KP
member	NO_KP
is	NO_KP
the	NO_KP
"current"	NO_KP
member	NO_KP
of	NO_KP
the	NO_KP
union.	NO_KP
therefore,	NO_KP
the	NO_KP
compiler	NO_KP
can't	NO_KP
know	NO_KP
when	NO_KP
it's	NO_KP
appropriate	NO_KP
to	NO_KP
call	NO_KP
constructors	BEGIN_KP
or	NO_KP
destructors	BEGIN_KP
on	NO_KP
the	NO_KP
union	BEGIN_KP
members.	NO_KP
still,	NO_KP
there	NO_KP
are	NO_KP
good	NO_KP
reasons	NO_KP
for	NO_KP
wanting	NO_KP
to	NO_KP
use	NO_KP
constructed	NO_KP
object	NO_KP
types	NO_KP
in	NO_KP
a	NO_KP
union.	NO_KP
for	NO_KP
example,	NO_KP
you	NO_KP
might	NO_KP
want	NO_KP
to	NO_KP
implement	NO_KP
a	NO_KP
scripting	BEGIN_KP
language	INSIDE_KP
with	NO_KP
a	NO_KP
single	NO_KP
variable	NO_KP
type	BEGIN_KP
that	NO_KP
can	NO_KP
either	NO_KP
be	NO_KP
an	NO_KP
integer,	NO_KP
a	NO_KP
string,	NO_KP
or	NO_KP
a	NO_KP
list.	NO_KP
a	NO_KP
union	BEGIN_KP
is	NO_KP
the	NO_KP
perfect	NO_KP
candidate	NO_KP
for	NO_KP
implementing	NO_KP
such	NO_KP
a	NO_KP
composite	NO_KP
type,	NO_KP
but	NO_KP
the	NO_KP
restriction	NO_KP
on	NO_KP
constructed	NO_KP
union	BEGIN_KP
members	INSIDE_KP
may	NO_KP
prevent	NO_KP
you	NO_KP
from	NO_KP
using	NO_KP
an	NO_KP
existing	NO_KP
string	NO_KP
or	NO_KP
list	NO_KP
class	NO_KP
(for	NO_KP
example,	NO_KP
from	NO_KP
the	NO_KP
stl)	NO_KP
to	NO_KP
provide	NO_KP
the	NO_KP
underlying	NO_KP
functionality.	NO_KP
luckily,	NO_KP
a	NO_KP
feature	NO_KP
of	NO_KP
c++	BEGIN_KP
called	NO_KP
placement	BEGIN_KP
new	INSIDE_KP
can	NO_KP
provide	NO_KP
a	NO_KP
workaround	NO_KP
